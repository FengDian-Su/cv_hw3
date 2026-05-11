"""
Visualization scripts for report.

1. plot_training_curve: plots train_loss and AP50 over epochs
2. visualize_predictions: draws predicted masks on original images

Usage:
    python visualize.py --mode curve \
        --history ../checkpoints_seed777/history.json \
        --output ../figures/training_curve.png

    python visualize.py --mode predict \
        --checkpoint ../checkpoints_seed777/best.pth \
        --data_root ../data/train \
        --output_dir ../figures/predictions
"""

import os
import argparse
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile as tiff
import torch
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────
# Training Curve
# ──────────────────────────────────────────────────────────────────────

def plot_training_curve(history_path, output_path):
    with open(history_path) as f:
        history = json.load(f)

    epochs      = [r["epoch"]      for r in history]
    train_loss  = [r["train_loss"] for r in history]
    eval_epochs = [r["epoch"] for r in history if r["ap50"] > 0]
    ap50_vals   = [r["ap50"]  for r in history if r["ap50"] > 0]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Train loss
    color_loss = "#2196F3"
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Train Loss", color=color_loss, fontsize=13)
    ax1.plot(epochs, train_loss, color=color_loss, linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    # AP50 on secondary axis
    ax2 = ax1.twinx()
    color_ap = "#F44336"
    ax2.set_ylabel("AP50 (Val)", color=color_ap, fontsize=13)
    ax2.plot(eval_epochs, ap50_vals, color=color_ap, linewidth=2,
             marker="o", markersize=6, label="Val AP50")
    ax2.tick_params(axis="y", labelcolor=color_ap)
    ax2.set_ylim(0, 0.75)

    # Mark best AP50
    best_ap   = max(ap50_vals)
    best_epoch = eval_epochs[ap50_vals.index(best_ap)]
    ax2.annotate(f"Best: {best_ap:.4f}\n(ep {best_epoch})",
                 xy=(best_epoch, best_ap),
                 xytext=(best_epoch + 3, best_ap - 0.05),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=10)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    plt.title("Training Curve (seed=777)", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curve → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Prediction Visualization
# ──────────────────────────────────────────────────────────────────────

# Color palette for 4 classes
CLASS_COLORS = {
    1: (255,  59,  48),   # red
    2: ( 52, 199,  89),   # green
    3: ( 10, 132, 255),   # blue
    4: (255, 204,   0),   # yellow
}
CLASS_NAMES = {1: "class1", 2: "class2", 3: "class3", 4: "class4"}

def overlay_masks_on_image(image_rgb, masks, labels, scores, alpha=0.45):
    """
    Draw coloured instance masks and bounding boxes on an RGB image.
    image_rgb : (H, W, 3) uint8
    masks     : (N, H, W) bool numpy
    labels    : (N,) int
    scores    : (N,) float
    """
    vis = image_rgb.copy().astype(np.float32)

    for mask, label, score in zip(masks, labels, scores):
        color = np.array(CLASS_COLORS.get(label, (200, 200, 200)),
                         dtype=np.float32)
        # Colour overlay
        for c in range(3):
            vis[:, :, c] = np.where(
                mask,
                vis[:, :, c] * (1 - alpha) + color[c] * alpha,
                vis[:, :, c],
            )
        # Contour (simple dilation - erosion trick)
        from scipy.ndimage import binary_dilation, binary_erosion
        contour = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
        for c in range(3):
            vis[:, :, c] = np.where(contour, color[c], vis[:, :, c])

    return np.clip(vis, 0, 255).astype(np.uint8)


@torch.no_grad()
def visualize_predictions(checkpoint_path, data_root, output_dir,
                          num_samples=4, score_threshold=0.07, seed=42):
    from dataset import CellDataset, build_train_val_split
    from train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = CellDataset.NUM_CLASSES + 1
    model = build_model(num_classes, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Pick samples from val set
    _, val_ids = build_train_val_split(data_root, val_ratio=0.15, seed=seed)
    random.seed(seed)
    selected = random.sample(val_ids, min(num_samples, len(val_ids)))

    os.makedirs(output_dir, exist_ok=True)

    for uid in selected:
        sample_dir = os.path.join(data_root, uid)
        img = tiff.imread(os.path.join(sample_dir, "image.tif"))
        img_rgb = img[:, :, :3]

        img_t = torch.as_tensor(img_rgb, dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        out    = model(img_t)[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks  = out["masks"].cpu().numpy()   # (N, 1, H, W)

        keep   = scores >= score_threshold
        scores = scores[keep]
        labels = labels[keep]
        masks  = (masks[keep, 0] >= 0.5)      # (N, H, W) bool

        # ── Build GT masks for comparison ─────────────────────────────
        gt_masks, gt_labels = [], []
        for cls_idx in range(1, 5):
            mp = os.path.join(sample_dir, f"class{cls_idx}.tif")
            if not os.path.exists(mp):
                continue
            lm = tiff.imread(mp).astype(np.int32)
            for inst_id in np.unique(lm):
                if inst_id == 0:
                    continue
                gt_masks.append(lm == inst_id)
                gt_labels.append(cls_idx)

        # ── Plot: original | GT | prediction ──────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        if gt_masks:
            gt_vis = overlay_masks_on_image(
                img_rgb,
                np.stack(gt_masks),
                np.array(gt_labels),
                np.ones(len(gt_labels)),
            )
            axes[1].imshow(gt_vis)
        else:
            axes[1].imshow(img_rgb)
        axes[1].set_title(f"Ground Truth ({len(gt_masks)} instances)", fontsize=12)
        axes[1].axis("off")

        if len(masks) > 0:
            pred_vis = overlay_masks_on_image(img_rgb, masks, labels, scores)
            axes[2].imshow(pred_vis)
        else:
            axes[2].imshow(img_rgb)
        axes[2].set_title(f"Prediction ({len(masks)} instances)", fontsize=12)
        axes[2].axis("off")

        # Legend
        patches = [mpatches.Patch(color=np.array(c)/255, label=CLASS_NAMES[i])
                   for i, c in CLASS_COLORS.items()]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(f"Sample: {uid[:8]}...", fontsize=11, y=1.01)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{uid[:8]}_vis.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["curve", "predict"], required=True)
    # curve
    p.add_argument("--history",     default="../checkpoints_seed777/history.json")
    # predict
    p.add_argument("--checkpoint",  default="../checkpoints_seed777/best.pth")
    p.add_argument("--data_root",   default="../data/train")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--score_threshold", type=float, default=0.07)
    # output
    p.add_argument("--output",      default="../figures/training_curve.png")
    p.add_argument("--output_dir",  default="../figures/predictions")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "curve":
        plot_training_curve(args.history, args.output)
    elif args.mode == "predict":
        visualize_predictions(
            args.checkpoint, args.data_root, args.output_dir,
            num_samples=args.num_samples,
            score_threshold=args.score_threshold,
        )


if __name__ == "__main__":
    main()