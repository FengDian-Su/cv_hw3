# """
# Inference script – generates COCO-format RLE submission JSON.

# Usage:
#     python predict.py --checkpoint ../checkpoints/best.pth \
#                       --test_dir ../data/test_release \
#                       --id_map ../data/test_image_name_to_ids.json \
#                       --output ../predictions.json \
#                       --score_threshold 0.5
# """

# import os
# import json
# import argparse

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from pycocotools import mask as mask_utils

# from dataset import CellTestDataset, CellDataset
# from train import build_model, collate_fn


# # ──────────────────────────────────────────────────────────────────────
# # RLE encoding helper
# # ──────────────────────────────────────────────────────────────────────

# def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
#     """
#     Encode a binary mask (H, W) bool/uint8 to COCO RLE format.
#     Uses pycocotools for compatibility with the official evaluator.
#     """
#     # pycocotools expects Fortran-contiguous uint8
#     mask_f = np.asfortranarray(binary_mask.astype(np.uint8))
#     rle = mask_utils.encode(mask_f)
#     rle["counts"] = rle["counts"].decode("utf-8")   # bytes → str for JSON
#     return rle


# # ──────────────────────────────────────────────────────────────────────
# # Inference
# # ──────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def run_inference(model, loader, device, score_threshold=0.5):
#     """
#     Run model on test set and collect predictions.

#     Returns:
#         List of dicts in COCO result format:
#         [
#           {
#             "image_id":    int,
#             "category_id": int,
#             "bbox":        [x, y, w, h],
#             "score":       float,
#             "segmentation": {"size": [H, W], "counts": str}
#           },
#           ...
#         ]
#     """
#     model.eval()
#     results = []

#     for batch_idx, (images, image_ids) in enumerate(loader):
#         images = [img.to(device) for img in images]

#         outputs = model(images)

#         for output, image_id in zip(outputs, image_ids):
#             scores  = output["scores"].cpu().numpy()
#             labels  = output["labels"].cpu().numpy()
#             boxes   = output["boxes"].cpu().numpy()     # (N, 4) x1y1x2y2
#             masks   = output["masks"].cpu().numpy()     # (N, 1, H, W) float in [0,1]

#             # Filter by score threshold
#             keep = scores >= score_threshold

#             for score, label, box, mask in zip(
#                 scores[keep], labels[keep], boxes[keep], masks[keep]
#             ):
#                 # Binarise soft mask at 0.5
#                 binary_mask = (mask[0] >= 0.5)          # (H, W) bool

#                 # COCO bbox format: [x, y, width, height]
#                 x1, y1, x2, y2 = box
#                 bbox_coco = [
#                     float(x1),
#                     float(y1),
#                     float(x2 - x1),
#                     float(y2 - y1),
#                 ]

#                 rle = encode_mask_to_rle(binary_mask)

#                 results.append({
#                     "image_id":    int(image_id),
#                     "category_id": int(label),
#                     "bbox":        bbox_coco,
#                     "score":       float(score),
#                     "segmentation": rle,
#                 })

#         if (batch_idx + 1) % 10 == 0:
#             print(f"  processed {batch_idx + 1}/{len(loader)} images ...")

#     return results


# # ──────────────────────────────────────────────────────────────────────
# # Main
# # ──────────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint",       default="../checkpoints/best.pth")
#     p.add_argument("--test_dir",         default="../data/test_release")
#     p.add_argument("--id_map",           default="../data/test_image_name_to_ids.json")
#     p.add_argument("--output",           default="../predictions.json")
#     p.add_argument("--score_threshold",  type=float, default=0.5)
#     p.add_argument("--num_workers",      type=int,   default=4)
#     return p.parse_args()


# def main():
#     args = parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # ── Model ─────────────────────────────────────────────────────────
#     num_classes = CellDataset.NUM_CLASSES + 1
#     model = build_model(num_classes, pretrained=False)

#     ckpt = torch.load(args.checkpoint, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     model.to(device)
#     print(f"Loaded checkpoint from {args.checkpoint}  (epoch {ckpt['epoch']})")

#     # ── Data ──────────────────────────────────────────────────────────
#     test_dataset = CellTestDataset(args.test_dir, args.id_map)
#     test_loader  = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=args.num_workers,
#         collate_fn=lambda x: (
#             [item[0] for item in x],
#             [item[1] for item in x],
#         ),
#     )
#     print(f"Test images: {len(test_dataset)}")

#     # ── Inference ─────────────────────────────────────────────────────
#     print(f"Running inference (score_threshold={args.score_threshold}) ...")
#     results = run_inference(model, test_loader, device, args.score_threshold)
#     print(f"Total predictions: {len(results)}")

#     # ── Save ──────────────────────────────────────────────────────────
#     os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
#     with open(args.output, "w") as f:
#         json.dump(results, f)
#     print(f"Saved predictions to {args.output}")


# if __name__ == "__main__":
#     main()
"""
Inference script – generates COCO-format RLE submission JSON.
Supports single model and multi-model ensemble.

Usage (single model):
    python predict.py \
        --checkpoints ../checkpoints/best.pth \
        --score_threshold 0.1 \
        --output ../predictions.json

Usage (ensemble):
    python predict.py \
        --checkpoints ../checkpoints_seed42/best.pth \
                      ../checkpoints_seed123/best.pth \
                      ../checkpoints_seed777/best.pth \
        --score_threshold 0.1 \
        --output ../predictions.json
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from pycocotools import mask as mask_utils

from dataset import CellTestDataset, CellDataset
from train import build_model, collate_fn


# ──────────────────────────────────────────────────────────────────────
# RLE encoding helper
# ──────────────────────────────────────────────────────────────────────

def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
    mask_f = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_f)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ──────────────────────────────────────────────────────────────────────
# Single model inference (one image at a time)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(model, image: torch.Tensor, device, score_threshold=0.1):
    """
    Run a single model on a single image.
    Returns (scores, labels, boxes, masks) as tensors on CPU.
    """
    img_in = image.unsqueeze(0).to(device)
    out    = model(img_in)[0]

    scores = out["scores"].cpu()
    labels = out["labels"].cpu()
    boxes  = out["boxes"].cpu()
    masks  = out["masks"].cpu()   # (N, 1, H, W)

    keep = scores >= score_threshold
    return scores[keep], labels[keep], boxes[keep], masks[keep]


# ──────────────────────────────────────────────────────────────────────
# Ensemble: merge predictions from multiple models with per-class NMS
# ──────────────────────────────────────────────────────────────────────

def ensemble_predictions(all_scores, all_labels, all_boxes, all_masks,
                         nms_threshold=0.5):
    """
    Merge predictions from N models and apply per-class mask-IoU NMS.

    Args:
        all_scores : list of (N_i,) tensors
        all_labels : list of (N_i,) tensors
        all_boxes  : list of (N_i, 4) tensors
        all_masks  : list of (N_i, 1, H, W) tensors
    Returns:
        scores, labels, boxes, masks (numpy arrays)
    """
    if len(all_scores) == 0:
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_f, empty_f.astype(np.int64), \
               np.zeros((0, 4), dtype=np.float32), \
               np.zeros((0,), dtype=object)

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    boxes  = torch.cat(all_boxes)
    masks  = torch.cat(all_masks)   # (N_total, 1, H, W)

    # Per-class box NMS to remove duplicates
    # Use a relaxed threshold because different models may predict
    # slightly different boxes for the same instance
    final_keep = []
    for cls in labels.unique():
        cls_idx = torch.where(labels == cls)[0]
        kept    = nms(boxes[cls_idx], scores[cls_idx], nms_threshold)
        final_keep.append(cls_idx[kept])

    if len(final_keep) == 0:
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_f, empty_f.astype(np.int64), \
               np.zeros((0, 4), dtype=np.float32), \
               np.zeros((0,), dtype=object)

    keep_idx = torch.cat(final_keep)

    return (scores[keep_idx].numpy(),
            labels[keep_idx].numpy(),
            boxes[keep_idx].numpy(),
            masks[keep_idx].numpy())


# ──────────────────────────────────────────────────────────────────────
# Full dataset inference
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(models, loader, device,
                  score_threshold=0.1, nms_threshold=0.5):
    """
    Run inference with one or more models.
    If multiple models are provided, their predictions are ensembled.
    """
    for m in models:
        m.eval()

    results = []

    for batch_idx, (images, image_ids) in enumerate(loader):
        for image, image_id in zip(images, image_ids):

            # Collect predictions from all models
            all_scores, all_labels, all_boxes, all_masks = [], [], [], []
            for model in models:
                s, l, b, m = predict_image(
                    model, image, device, score_threshold
                )
                if len(s) > 0:
                    all_scores.append(s)
                    all_labels.append(l)
                    all_boxes.append(b)
                    all_masks.append(m)

            if len(all_scores) == 0:
                continue

            # Ensemble / NMS
            scores, labels, boxes, masks = ensemble_predictions(
                all_scores, all_labels, all_boxes, all_masks,
                nms_threshold=nms_threshold,
            )

            for score, label, box, mask in zip(scores, labels, boxes, masks):
                binary_mask = (mask[0] >= 0.5)
                x1, y1, x2, y2 = box
                results.append({
                    "image_id":    int(image_id),
                    "category_id": int(label),
                    "bbox":        [float(x1), float(y1),
                                    float(x2 - x1), float(y2 - y1)],
                    "score":       float(score),
                    "segmentation": encode_mask_to_rle(binary_mask),
                })

        if (batch_idx + 1) % 10 == 0:
            print(f"  processed {batch_idx + 1}/{len(loader)} images ...")

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints",     nargs="+",
                   default=["../checkpoints/best.pth"],
                   help="One or more checkpoint paths for ensemble")
    p.add_argument("--test_dir",        default="../data/test_release")
    p.add_argument("--id_map",          default="../data/test_image_name_to_ids.json")
    p.add_argument("--output",          default="../predictions.json")
    p.add_argument("--score_threshold", type=float, default=0.1)
    p.add_argument("--nms_threshold",   type=float, default=0.5)
    p.add_argument("--num_workers",     type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load models ───────────────────────────────────────────────────
    num_classes = CellDataset.NUM_CLASSES + 1
    models = []
    for ckpt_path in args.checkpoints:
        model = build_model(num_classes, pretrained=False)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        print(f"Loaded {ckpt_path}  (epoch {ckpt['epoch']})")
        models.append(model)

    print(f"Ensemble size: {len(models)} model(s)")

    # ── Data ──────────────────────────────────────────────────────────
    test_dataset = CellTestDataset(args.test_dir, args.id_map)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: (
            [item[0] for item in x],
            [item[1] for item in x],
        ),
    )
    print(f"Test images: {len(test_dataset)}")

    # ── Inference ─────────────────────────────────────────────────────
    print(f"Running inference "
          f"(score_threshold={args.score_threshold}, "
          f"nms_threshold={args.nms_threshold}) ...")
    results = run_inference(
        models, test_loader, device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    print(f"Total predictions: {len(results)}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
