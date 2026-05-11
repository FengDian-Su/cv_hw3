"""
Training script
  - Mask R-CNN with ResNet-101 + FPN backbone
  - ImageNet pretrained weights
  - Multi-scale training: min_size=(300,400,500,600), max_size=700
  - SGD optimizer, cosine LR schedule
  - Gradient accumulation to avoid OOM

Usage:
    CUDA_VISIBLE_DEVICES=1 python train.py
"""

import os
import time
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CellDataset, build_train_val_split
from transforms import get_train_transforms


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Mask R-CNN with ResNet-101 + FPN.

    Key changes:
      - min_size=(300,400,500,600), max_size=700
          → Multi-scale training to generalise to test images of varying sizes.
          → Also reduces VRAM vs the default min_size=800/max_size=1333.
      - box_detections_per_img=300 for dense cell scenes.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT if pretrained else None,
        trainable_layers=5,
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_detections_per_img=300,
        min_size=(300, 400, 500, 600),
        max_size=700,
        # min_size=(200, 300, 400, 500, 600),
        # max_size=700,
    )

    return model


# ──────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ──────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch,
                    accum_steps=2, print_freq=10):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values()) / accum_steps

        losses.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses.item() * accum_steps

        if (i + 1) % print_freq == 0 or (i + 1) == n_batches:
            loss_str = "  ".join(
                f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
            )
            print(f"  [Epoch {epoch}  {i+1}/{n_batches}]  {loss_str}  "
                  f"total: {losses.item() * accum_steps:.4f}")

    return total_loss / n_batches


# ──────────────────────────────────────────────────────────────────────
# Val AP50
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ap50(model, data_root, val_ids, device, score_threshold=0.05):
    import tifffile as tiff
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()

    # ── Build GT ──────────────────────────────────────────────────────
    images_info = []
    annotations = []
    ann_id = 1

    for img_idx, uid in enumerate(val_ids):
        image_id   = img_idx + 1
        sample_dir = os.path.join(data_root, uid)
        img = tiff.imread(os.path.join(sample_dir, "image.tif"))
        H, W = img.shape[0], img.shape[1]
        images_info.append({"id": image_id, "height": H, "width": W,
                             "file_name": uid + ".tif"})

        for cls_idx in range(1, 5):
            mask_path = os.path.join(sample_dir, f"class{cls_idx}.tif")
            if not os.path.exists(mask_path):
                continue
            label_map = tiff.imread(mask_path).astype(np.int32)
            for inst_id in np.unique(label_map):
                if inst_id == 0:
                    continue
                bm = (label_map == inst_id).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(bm))
                ys, xs = np.where(bm)
                annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cls_idx,
                    "segmentation": {"size": rle["size"],
                                     "counts": rle["counts"].decode()},
                    "bbox": [int(xs.min()), int(ys.min()),
                             int(xs.max() - xs.min() + 1),
                             int(ys.max() - ys.min() + 1)],
                    "area": int(bm.sum()), "iscrowd": 0,
                })
                ann_id += 1

    categories = [{"id": i, "name": f"class{i}"} for i in range(1, 5)]
    gt_dict = {"images": images_info, "annotations": annotations,
               "categories": categories}

    # ── Predict ───────────────────────────────────────────────────────
    results = []
    for img_idx, uid in enumerate(val_ids):
        image_id = img_idx + 1
        img = tiff.imread(os.path.join(data_root, uid, "image.tif"))
        img_t = torch.as_tensor(
            img[:, :, :3], dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        out = model(img_t)[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        boxes  = out["boxes"].cpu().numpy()
        masks  = out["masks"].cpu().numpy()

        for score, label, box, mask in zip(scores, labels, boxes, masks):
            if score < score_threshold:
                continue
            bm = (mask[0] >= 0.5).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(bm))
            x1, y1, x2, y2 = box
            results.append({
                "image_id":    image_id,
                "category_id": int(label),
                "bbox":        [float(x1), float(y1),
                                float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
                "segmentation": {"size": rle["size"],
                                 "counts": rle["counts"].decode()},
            })

    if len(results) == 0:
        print("  Warning: no predictions, AP50 = 0")
        return 0.0

    # ── COCOEval ──────────────────────────────────────────────────────
    import tempfile, json as _json, io, contextlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _json.dump(gt_dict, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _json.dump(results, f)
        pred_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.evaluate()
    ev.accumulate()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    print(buf.getvalue(), end="")

    os.unlink(gt_path)
    os.unlink(pred_path)
    return float(ev.stats[1])   # AP50


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default="../data/train")
    p.add_argument("--output_dir",   default="../checkpoints")
    p.add_argument("--num_epochs",   type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--accum_steps",  type=int,   default=2,
                   help="gradient accumulation steps; effective batch = batch_size * accum_steps")
    p.add_argument("--lr",           type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--val_ratio",    type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--eval_freq",    type=int,   default=5,
                   help="run COCOEval every N epochs")
    p.add_argument("--resume",       default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────
    # 原本
    # train_ids, val_ids = build_train_val_split(
    #     args.data_root, val_ratio=args.val_ratio, seed=args.seed
    # )

    # 改成
    if args.val_ratio == 0:
        all_ids = sorted([
            d for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ])
        train_ids, val_ids = all_ids, []
    else:
        train_ids, val_ids = build_train_val_split(
            args.data_root, val_ratio=args.val_ratio, seed=args.seed
        )
    print(f"Train samples: {len(train_ids)}  Val samples: {len(val_ids)}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")

    train_dataset = CellDataset(
        args.data_root, train_ids, transforms=get_train_transforms()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    num_classes = CellDataset.NUM_CLASSES + 1
    model = build_model(num_classes, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.1f}M")
    print(f"min_size={model.transform.min_size}  max_size={model.transform.max_size}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_ap50   = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ap50   = ckpt.get("best_ap50", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Training Loop ─────────────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            accum_steps=args.accum_steps,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.num_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"lr={lr_now:.6f}  time={elapsed:.1f}s")

        # Save latest
        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_ap50": best_ap50,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))

        # Periodic COCOEval
        ap50 = 0.0
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            # 原本
            # print(f"  [Eval @ epoch {epoch}]")
            # ap50 = evaluate_ap50(model, args.data_root, val_ids, device)
            # print(f"  >>> AP50 = {ap50:.4f}")

            # if ap50 > best_ap50:
            #     best_ap50 = ap50
            #     torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            #     print(f"  ✓ Saved best model  (AP50={ap50:.4f})")
            
            # 修改
            if len(val_ids) > 0:
                print(f"  [Eval @ epoch {epoch}]")
                ap50 = evaluate_ap50(model, args.data_root, val_ids, device)
                print(f"  >>> AP50 = {ap50:.4f}")
                if ap50 > best_ap50:
                    best_ap50 = ap50
                    torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
                    print(f"  ✓ Saved best model  (AP50={ap50:.4f})")
            else:
                # No val set, save every eval_freq epochs and treat last as best
                torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
                print(f"  ✓ Saved model (no val set, epoch {epoch})")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "ap50":       ap50,
            "lr":         lr_now,
        })

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best AP50 = {best_ap50:.4f}")


if __name__ == "__main__":
    main()
