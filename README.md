# CSCI 611 — Food Image Classification Benchmark

Benchmarks three CNN architectures on a 34-class food image dataset (~24k images).
The goal is to compare accuracy, inference speed, and model size across a heavy
(ResNet50), intermediate (EfficientNet-B0), and light (MobileNetV2) architecture,
all fine-tuned from ImageNet pretrained weights with Optuna hyperparameter tuning.

---

## Results

| Model | Params | Size (MB) | Val Acc | Test Top-1 | Test Top-5 | Latency (ms) | Throughput (img/s) |
|---|---|---|---|---|---|---|---|
| ResNet50 | 23.6M | 94.6 | 92.3% | **92.7%** | 99.3% | 2.74 | 1,477 |
| EfficientNet-B0 | 4.1M | 16.5 | 92.2% | 92.2% | **99.5%** | 3.71 | 3,270 |
| MobileNetV2 | 2.3M | **9.3** | 90.4% | 91.1% | 99.3% | **2.33** | **4,030** |

Benchmarked on GPU (CUDA). EfficientNet-B0 matches ResNet50 accuracy at 1/6th the
size and 2× the throughput. MobileNetV2 is the fastest and smallest at a ~1.6%
accuracy cost.

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

The dataset lives at `../dataset` (one level above the repo root), organized as:

```
dataset/
  Baked Potato/
  Crispy Chicken/
  ...  (34 classes total)
```

| Split | Images |
|---|---|
| Train | 19,092 |
| Val | 2,377 |
| Test | 2,404 |
| **Total** | **23,873** |

The split manifest (`configs/data_split.yaml`) is already committed. To regenerate
it from a different dataset root:

```bash
python scripts/generate_split_manifest.py \
    --dataset-root ../dataset \
    --output configs/data_split.yaml
```

---

## Workflow

### 1. Train each architecture

Each script runs an Optuna sweep and saves the best checkpoint to
`outputs/checkpoints/<arch>/trial_<N>/`.

```bash
python -m src.train_resnet
python -m src.train_efficientnet
python -m src.train_mobilenet
```

Best hyperparameters found per arch:

| Model | Optimizer | LR | Batch | Dropout | Notes |
|---|---|---|---|---|---|
| ResNet50 | AdamW | 4.3e-5 | 16 | 0.13 | label_smoothing=0.098 |
| EfficientNet-B0 | SGD | 8.1e-3 | 64 | 0.47 | — |
| MobileNetV2 | SGD | 2.2e-3 | 32 | 0.20 | backbone unfrozen |

### 2. Evaluate on the test set

```bash
python -m src.evaluate --arch resnet50
python -m src.evaluate --arch efficientnet_b0
python -m src.evaluate --arch mobilenet_v2
```

Writes per-class accuracy, confusion matrix, and metrics JSON to `outputs/eval/<arch>/`.

### 3. Benchmark inference speed

```bash
python scripts/benchmark_speed.py
```

Writes latency and throughput results to `outputs/speed_benchmark/`.

### 4. Grad-CAM visualization

Compare all three models on the same image, showing early vs. late convolutional
layer attribution side by side:

```bash
python scripts/gradcam_compare.py --image ../dataset/Taco/some_image.jpg
```

Output: `outputs/gradcam_compare.png`

To save to a different path:

```bash
python scripts/gradcam_compare.py \
    --image ../dataset/Taco/some_image.jpg \
    --output outputs/gradcam_taco.png
```

Single-arch Grad-CAM (picks the best checkpoint automatically via the summary JSON):

```bash
python -m src.gradcam \
    --arch resnet50 \
    --checkpoint outputs/checkpoints/resnet50/trial_18/resnet50_trial18_best.pt \
    --image ../dataset/Taco/some_image.jpg
```

### 5. Generate report figures

```bash
python scripts/generate_figures.py
```

Writes accuracy plots, Optuna optimization history, and a comparison table to
`outputs/report/`.

### 6. Analyze confusion patterns

```bash
python scripts/analyze_confusions.py
```

Writes top confusion pairs per arch to `outputs/eval/<arch>/top_confusions.csv`.

---

## Project structure

```
CSCI611-FinalProject/
├── configs/
│   └── data_split.yaml          # train/val/test manifest (seed=42)
├── outputs/
│   ├── checkpoints/             # per-trial .pt weights (gitignored except best)
│   ├── eval/                    # per-arch confusion matrices & metrics
│   ├── report/                  # figures and comparison_table.csv
│   ├── speed_benchmark/         # latency / throughput JSON + plots
│   ├── optuna_studies/          # Optuna .db files for resuming sweeps
│   ├── *_best_summary.json      # winning trial number + hyperparams per arch
│   └── gradcam_compare*.png     # Grad-CAM comparison figures
├── scripts/
│   ├── analyze_confusions.py    # confusion matrix analysis
│   ├── benchmark_speed.py       # inference speed benchmark
│   ├── generate_split_manifest.py
│   ├── gradcam_compare.py       # early vs. late layer Grad-CAM across all arches
│   └── generate_figures.py      # report figures and tables
└── src/
    ├── data_prep.py             # dataset loading, transforms, manifest parsing
    ├── evaluate.py              # test-set evaluation
    ├── gradcam.py               # Grad-CAM implementation
    ├── models.py                # model factories + Grad-CAM target layer registry
    ├── train.py                 # shared training loop
    ├── train_efficientnet.py    # EfficientNet-B0 Optuna sweep
    ├── train_mobilenet.py       # MobileNetV2 Optuna sweep
    ├── train_resnet.py          # ResNet50 Optuna sweep
    └── tune_optuna.py           # shared Optuna objective + trial logic
```
