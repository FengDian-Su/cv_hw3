## Cell Instance Segmentation

### Introduction

This project implements an instance segmentation pipeline for colored medical cell images, targeting four cell types (class1–class4). The model is based on Mask R-CNN with a ResNet-101 + FPN backbone, trained with multi-scale augmentation and ensembled across multiple random seeds for improved generalization.

---

### Environment Setup

#### Requirements

```bash
pip install torch torchvision
pip install tifffile
pip install pycocotools
pip install opencv-python-headless
pip install imagecodecs
pip install scipy
pip install matplotlib
```

#### Project Structure

```
cv_hw3/
├── code/
│   ├── dataset.py        # Dataset loading and preprocessing
│   ├── transforms.py     # Data augmentation
│   ├── train.py          # Model definition and training
│   ├── predict.py        # Inference and ensemble
│   └── visualize.py      # Training curve and prediction visualization
├── data/
│   ├── train/
│   │   └── {uuid}/
│   │       ├── image.tif
│   │       ├── class1.tif
│   │       ├── class2.tif
│   │       ├── class3.tif
│   │       └── class4.tif
│   ├── test_release/
│   │   └── {uuid}.tif
│   └── test_image_name_to_ids.json
├── checkpoints/          # Saved model checkpoints
└── figures/              # Visualization outputs
```

---

### Usage

#### 1. Train

Train the model with default settings (50 epochs, seed=42):

```bash
cd code
python3 train.py \
    --data_root ../data/train \
    --output_dir ../checkpoints \
    --num_epochs 50 \
    --batch_size 1
```

To train with a different seed (for ensemble):

```bash
python3 train.py \
    --data_root ../data/train \
    --output_dir ../checkpoints_seed777 \
    --num_epochs 50 \
    --batch_size 1 \
    --seed 777
```

To resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --resume ../checkpoints/last.pth \
    --num_epochs 50 \
    --batch_size 1
```

#### 2. Evaluate (Local AP50)

Compute AP50 on the validation set using COCOEval:

```bash
python3 evaluate.py \
    --checkpoint ../checkpoints/best.pth \
    --data_root ../data/train \
    --output_dir ../eval_output
```

#### 3. Predict (Single Model)

Generate predictions for the test set:

```bash
python3 predict.py \
    --checkpoints ../checkpoints/best.pth \
    --test_dir ../data/test_release \
    --id_map ../data/test_image_name_to_ids.json \
    --score_threshold 0.07 \
    --output ../predictions.json
```

#### 4. Predict (Ensemble)

Combine predictions from multiple models:

```bash
python3 predict.py \
    --checkpoints ../checkpoints/best.pth \
                  ../checkpoints_seed777/best.pth \
    --test_dir ../data/test_release \
    --id_map ../data/test_image_name_to_ids.json \
    --score_threshold 0.07 \
    --nms_threshold 0.5 \
    --output ../predictions_ensemble.json
```

#### 5. Visualize

Plot the training curve:

```bash
python3 visualize.py --mode curve \
    --history ../checkpoints/history.json \
    --output ../figures/training_curve.png
```

Visualize predictions on validation samples:

```bash
python3 visualize.py --mode predict \
    --checkpoint ../checkpoints/best.pth \
    --data_root ../data/train \
    --num_samples 4 \
    --output_dir ../figures/predictions
```

---

### Performance Snapshot

The final submission uses an ensemble of two Mask R-CNN models (ResNet-101 + FPN) trained with different random seeds and multi-scale training (`min_size=(300, 400, 500, 600)`). Predictions are merged with per-class NMS and filtered at a score threshold of 0.07.

| Configuration | AP50 |
|---------------|------|
| Single model (seed=42) | 0.4995 |
| Single model (seed=777) | 0.5045 |
| **Ensemble (seed=42 + seed=777)** | **0.5286** |

![leaderboard.png](leaderboard.png)
