<div align="center">

  <h1>Image Classification Pipeline</h1>

</div>

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Test](#test)
  - [CLI Reference](#cli-reference)
- [Configuration](#configuration)
  - [Supported Models](#supported-models)
  - [Supported Optimizers & Schedulers](#supported-optimizers--schedulers)
  - [Config Reference](#config-reference)
- [Output](#output)
- [Experiments](#experiments)


## About

A PyTorch framework for training and benchmarking image classification models. Supports multiple architectures, configurable training pipelines, and automatic metric logging and visualization — all driven by a single YAML config file. For more information about the proccess of making the project and technical details check `doc/design.md` (to be added).

### Project Structure

```
project/
├── main.py
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── architectural_styles
│   │   ├──raw/
│   │   │   ├── Achaemenid architecture/
│   │   │   ├── American craftsman style/
│   │   │   └── .../
│   │   └── processed/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── scripts/
│       ├──prepare_data.py
│       └──stats_data.py
├── experiments/
│   └── <experiment_name>/
│       ├── checkpoints/
│       ├── config.yaml
│       ├── history.json
│       └── visualisation/
│           ├── train/
│           └── test/
├── src/
│   ├── engine.py
│   ├── models/
│   │   └── architectures.py
│   ├── training/
│   │   ├── train.py
│   │   └── test.py
│   └── utils/
│       ├── data.py
│       ├── logger.py
│       ├── metrics.py
│       ├── optim.py
│       ├── plotting.py
│       └── seed.py
└── doc/
    ├── config.md
    ├── design.md
    └── experiments.md
```

Data must follow the `ImageFolder` convention: each subdirectory under `train/`, `val/`, and `test/`  represents one class. Use `data/scripts/prepare_data.py` to split raw data and `data/scripts/stats_data.py` to compute per-channel mean and std for your dataset.

---

## Installation
 
```bash
pip install -r requirements.txt
pip install -e .
```
 
> PyTorch and torchvision are listed with a CUDA 13.0 build. Install the variant matching your environment from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### Train

```bash
# General
python main.py --config path/to/config.yaml --mode train
 
# Example
python main.py --config experiments/testing/config.yaml --mode train
```
 
To resume from a checkpoint and restore metric history:
 
```bash
python main.py --config experiments/testing/config.yaml --mode train \
  --checkpoint experiments/testing/checkpoints/epoch_10.pth \
  --history experiments/testing/history.json
```

### Test
 
```bash
# General
python main.py --config path/to/config.yaml --mode test --checkpoint path/to/checkpoint.pth
 
# Example
python main.py --config experiments/testing/config.yaml --mode test \
  --checkpoint experiments/testing/checkpoints/best_model.pth
```

`--checkpoint` is required in test mode.

### CLI Reference
 
| Flag | Short | Default | Description |
|---|---|---|---|
| `--config` | `-cfg` | `experiments/testing/config.yaml` | Path to YAML config |
| `--mode` | `-m` | `train` | `train` or `test` |
| `--checkpoint` | `-ckpt` | `None` | Checkpoint path to resume from (required for test)|
| `--history` | `-hist` | `None` | History JSON from a previous run |
| `--log-level` | `-l` | from config | Override log level |
| `--log-file` | | from config | Override log file path |

---

## Configuration

### Supported Models
 
| Config name | Architecture |
|---|---|
| `vgg16` | VGG-16 |
| `resnet50` | ResNet-50 |
| `efficientnet_b0` | EfficientNet-B0 |
| `convnext_tiny` | ConvNeXt-Tiny |
 
All models load ImageNet pretrained weights when `pretrained: true` and replace the final classification head to match `num_classes`.

### Supported Optimizers & Schedulers
 
**Optimizers:** `sgd`, `adam`, `adamw`
 
**LR Schedulers:** `cosine` (CosineAnnealingLR), `step` (StepLR) — optional, omit `scheduler` key to disable.
 
See `doc/config.md` for the full list of parameters for each optimizer and scheduler.

### Config Reference
  
```yaml
seed: 42
log_level: "INFO"
log_file: "experiments/testing/testing.log" # <experiment>/<name>.log
save_dir: "experiments/testing"

data:
  train_dir: "data/architectural_styles/processed/train"
  val_dir: "data/architectural_styles/processed/val"
  test_dir: "data/architectural_styles/processed/test"
  mean: [0.4963, 0.4963, 0.4894]    # per-channel, compute with data/scripts/stats_data.py
  std:  [0.2304, 0.2314, 0.2542]
  batch_size: 32
  num_workers: 4
  image_size: 224
  resize_size: 256                  # resize before center-cropping in val/test pipeline
  drop_last: true
  augmentations:                    # applied to train split only; omit key to disable
    - name: RandomHorizontalFlip
      params:
        p: 0.5
    - name: RandomResizedCrop
      params:
        size: 224

model:
  name: "resnet50"
  num_classes: 25
  pretrained: true
 
train:
  epochs: 50
  loss_goal: 0.1        # stop early if val loss reaches this value
  patience: 5           # early stopping patience (epochs)
  checkpoint_interval: 10
  optimizer:
    name: "sgd"
    lr: 0.001
    momentum: 0.9
    scheduler:
      name: cosine
      T_max: 50         # typically set to total training epochs
  advanced_metrics: true
  plotting:
    enabled: true
    advanced_metrics: true
 
test:
  advanced_metrics: true
  plotting:
    enabled: true
```
 
Full config documentation for all keys, types, defaults, and edge cases is in `doc/config.md`.

---

## Output
 
After a run, `save_dir` contains:
 
- `history.json` — per-epoch train/val metrics
- `*.log` — training log
- `test_metrics.json` — per-class metrics and confusion matrix for test dataset
- `visualisation/train/` — loss curve, accuracy curve, confusion matrix, macro metrics (if enabled)
- `visualisation/test/` — confusion matrix, per-class metrics (if enabled)
- `checkpoints/` — Model checkpoints saved every `checkpoint_interval` epochs along with final and best model

--- 

## Experiments
 
See `doc/experiments.md`(to be added) for a log of all runs with notes on configurations and results.
