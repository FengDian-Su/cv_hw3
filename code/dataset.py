import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import cv2
from PIL import Image
import torchvision.transforms.functional as F


class CellDataset(Dataset):
    """
    Reads the cell instance segmentation dataset.

    Each training sample lives in:
        data/train/{uuid}/
            image.tif        – (H, W, 4) uint8, RGBA  → we drop channel 3
            class1.tif       – (H, W) float64, pixel value = instance ID (0 = bg)
            class2.tif       – same
            class3.tif       – same  (may not exist)
            class4.tif       – same  (may not exist)

    Returns a dict compatible with torchvision Mask R-CNN:
        image   : FloatTensor (3, H, W), range [0, 1]
        target  : dict with keys
                    boxes    – FloatTensor (N, 4)  [x1, y1, x2, y2]
                    labels   – Int64Tensor  (N,)   1-indexed class id
                    masks    – BoolTensor   (N, H, W)
                    image_id – Int64Tensor  scalar
    """

    NUM_CLASSES = 4  # class1 … class4

    def __init__(self, root, sample_ids, transforms=None):
        """
        Args:
            root        : path to data/train/
            sample_ids  : list of uuid folder names to include
            transforms  : callable applied to (image, target), optional
        """
        self.root = root
        self.sample_ids = sample_ids
        self.transforms = transforms

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sample_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        uid = self.sample_ids[idx]
        sample_dir = os.path.join(self.root, uid)

        # ── 1. Load image (H, W, 4) → RGB (H, W, 3) ──────────────────
        image = tiff.imread(os.path.join(sample_dir, "image.tif"))
        image = image[:, :, :3]                        # drop alpha
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        H, W = image.shape[1], image.shape[2]

        # ── 2. Load masks for all classes ─────────────────────────────
        all_masks  = []   # binary (H, W) bool
        all_labels = []   # int class id (1-indexed)

        for cls_idx in range(1, self.NUM_CLASSES + 1):
            mask_path = os.path.join(sample_dir, f"class{cls_idx}.tif")
            if not os.path.exists(mask_path):
                continue

            label_map = tiff.imread(mask_path)          # float64, values = instance IDs
            label_map = label_map.astype(np.int32)

            instance_ids = np.unique(label_map)
            instance_ids = instance_ids[instance_ids != 0]   # remove background

            for inst_id in instance_ids:
                binary_mask = (label_map == inst_id)    # bool (H, W)
                all_masks.append(binary_mask)
                all_labels.append(cls_idx)

        # ── 3. Build target dict ───────────────────────────────────────
        if len(all_masks) == 0:
            # Edge case: no annotations in this sample
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            masks   = torch.zeros((0, H, W), dtype=torch.bool)
        else:
            masks_np = np.stack(all_masks, axis=0)          # (N, H, W) bool
            masks    = torch.as_tensor(masks_np, dtype=torch.bool)
            labels   = torch.as_tensor(all_labels, dtype=torch.int64)

            # Bounding boxes from masks
            boxes = masks_to_boxes(masks)                   # (N, 4)

            # Filter out degenerate boxes (w or h == 0)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes  = boxes[keep]
            labels = labels[keep]
            masks  = masks[keep]

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor(idx, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# ──────────────────────────────────────────────────────────────────────
# Test dataset (no masks)
# ──────────────────────────────────────────────────────────────────────

class CellTestDataset(Dataset):
    """
    Loads test images from data/test_release/.
    Also reads test_image_name_to_ids.json so we can attach the correct
    COCO image_id to every prediction.
    """

    def __init__(self, test_dir, id_map_path):
        """
        Args:
            test_dir    : path to data/test_release/
            id_map_path : path to test_image_name_to_ids.json
        """
        self.test_dir = test_dir

        with open(id_map_path, "r") as f:
            entries = json.load(f)

        # Build {filename → id} mapping
        self.fname_to_id = {e["file_name"]: e["id"] for e in entries}

        # Keep only files that actually exist on disk
        self.entries = [
            e for e in entries
            if os.path.exists(os.path.join(test_dir, e["file_name"]))
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        fpath = os.path.join(self.test_dir, entry["file_name"])

        image = tiff.imread(fpath)
        image = image[:, :, :3]                         # drop alpha
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, entry["id"]   # (tensor, coco_image_id)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding boxes from binary masks.

    Args:
        masks: BoolTensor of shape (N, H, W)
    Returns:
        FloatTensor of shape (N, 4) with [x1, y1, x2, y2]
    """
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32)

    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if len(xs) == 0:
            continue
        boxes[i] = torch.tensor(
            [xs.min(), ys.min(), xs.max(), ys.max()],
            dtype=torch.float32,
        )
    return boxes


def build_train_val_split(train_root, val_ratio=0.15, seed=42):
    """
    Scans train_root for uuid subdirectories and returns
    (train_ids, val_ids) lists.
    """
    all_ids = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_ids))

    n_val = max(1, int(len(all_ids) * val_ratio))
    val_indices   = indices[:n_val]
    train_indices = indices[n_val:]

    train_ids = [all_ids[i] for i in sorted(train_indices)]
    val_ids   = [all_ids[i] for i in sorted(val_indices)]

    return train_ids, val_ids


# ──────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_root = "data/train"
    train_ids, val_ids = build_train_val_split(train_root)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    ds = CellDataset(train_root, train_ids)
    img, tgt = ds[0]
    print(f"image shape : {img.shape}")
    print(f"num instances: {len(tgt['labels'])}")
    print(f"labels       : {tgt['labels'][:10]}")
    print(f"boxes shape  : {tgt['boxes'].shape}")
    print(f"masks shape  : {tgt['masks'].shape}")