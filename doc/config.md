# Configuration Reference

All behaviour is controlled by a single YAML file passed via `--config`. This document describes every available key, its type, default value, and valid options.

## Table of Contents

- [Top-level keys](#top-level-keys)
- [data](#data)
  - [Augmentations](#augmentations)
- [model](#model)
- [train](#train)
  - [optimizer](#optimizer)
  - [scheduler](#scheduler)
- [test](#test)

---

## Top-level keys

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `None` | Global random seed for reproducibility. Passed to PyTorch, NumPy, and dataloader workers. Omit to run without a fixed seed. |
| `log_level` | `str` | `"INFO"` | Logging verbosity. Valid values: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. Can be overridden at runtime with `--log-level`. |
| `log_file` | `str` | `None` | Path to write the log file. If omitted, logs go to stdout only. Can be overridden at runtime with `--log-file`. |
| `save_dir` | `str` | `None` | Root directory for all outputs (checkpoints, history, visualisations, test metrics). If omitted, nothing is written to disk and the best model state is kept in memory only. |

---

## data

Controls dataset paths, preprocessing, and the dataloader.

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `train_dir` | `str` | — | Yes | Path to the training split. Must follow `ImageFolder` convention (one subdirectory per class). |
| `val_dir` | `str` | — | Yes | Path to the validation split. Same convention as `train_dir`. |
| `test_dir` | `str` | — | Yes | Path to the test split. Same convention as `train_dir`. |
| `mean` | `list[float]` | — | Yes | Per-channel mean for normalisation, in RGB order. Compute on your training set with `data/scripts/stats_data.py`. |
| `std` | `list[float]` | — | Yes | Per-channel standard deviation for normalisation, in RGB order. Compute on your training set with `data/scripts/stats_data.py`. |
| `batch_size` | `int` | — | Yes | Number of samples per batch for all splits. |
| `num_workers` | `int` | `0` | No | Number of subprocesses for data loading. Set to `0` to load in the main process. |
| `image_size` | `int` | — | Yes | Final image size in pixels after all transforms. Used as the crop size in the val/test pipeline and in the train pipeline if no `augmentations` is added. |
| `resize_size` | `int` | — | Yes | Size to resize the shorter edge to before centre-cropping, in the val/test pipeline and in the train pipeline if no `augmentations` is added. Should be larger than `image_size` (e.g. 256 when `image_size` is 224). |
| `drop_last` | `bool` | `false` | No | If `true`, drops the last incomplete batch in all splits. Useful when batch normalisation layers are present. |
| `augmentations` | `list[dict]` | `None` | No | List of augmentations applied to the **train split only**. Omit the key entirely to disable augmentation. See [Augmentations](#augmentations). |

**Transform pipelines:**

- **Train (with augmentations):** augmentations → `ToTensor` → `Normalize`
- **Train (without augmentations):** `Resize(resize_size)` → `CenterCrop(image_size)` → `ToTensor` → `Normalize`
- **Val / Test:** `Resize(resize_size)` → `CenterCrop(image_size)` → `ToTensor` → `Normalize`

### Augmentations

Each entry in the `augmentations` list requires a `name` key and an optional `params` dict. Augmentations are applied in the order they are listed.

```yaml
augmentations:
  - name: RandomHorizontalFlip
    params:
      p: 0.5
  - name: ColorJitter
    params:
      brightness: 0.2
      contrast: 0.2
```

Supported augmentations:

| Name | Parameter | Type | Default | Description |
|---|---|---|---|---|
| `RandomResizedCrop` | `size` | `int` | — | Crop a random portion of the image and resize it to `size`. Uses torchvision defaults for `scale` and `ratio` if it is not specified. |
| `RandomHorizontalFlip` | `p` | `float` | `0.5` | Probability of flipping the image horizontally. |
| `RandomVerticalFlip` | `p` | `float` | `0.5` | Probability of flipping the image vertically. |
| `ColorJitter` | `brightness` | `float` | `0` | How much to jitter brightness. `0` disables it. |
| | `contrast` | `float` | `0` | How much to jitter contrast. |
| | `saturation` | `float` | `0` | How much to jitter saturation. |
| | `hue` | `float` | `0` | How much to jitter hue. Must be in `[0, 0.5]`. |
| `RandomRotation` | `degrees` | `float` | — | Range of degrees for random rotation `(-degrees, +degrees)`. |
| `RandomGrayscale` | `p` | `float` | `0.1` | Probability of converting the image to grayscale. Output still has 3 channels. |

---

## model

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | `str` | — | Yes | Architecture to use. See supported values below. |
| `num_classes` | `int` | — | Yes | Number of output classes. The final classification layer is replaced to match this value. |
| `pretrained` | `bool` | `false` | No | If `true`, loads ImageNet pretrained weights for all layers except the final head. |

Supported model names:

| Value | Architecture |
|---|---|
| `vgg16` | VGG-16 |
| `resnet50` | ResNet-50 |
| `efficientnet_b0` | EfficientNet-B0 |
| `convnext_tiny` | ConvNeXt-Tiny |

---

## train

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `epochs` | `int` | — | Yes | Maximum number of training epochs. |
| `optimizer` | `dict` | — | Yes | Optimizer configuration. See [optimizer](#optimizer). |
| `checkpoint_interval` | `int` | `10` | No | Save a checkpoint every this many epochs to `<save_dir>/checkpoints/checkpoint_epoch_{N}`. |
| `patience` | `int` | `None` | No | Early stopping patience. Training stops if validation loss does not improve for this many consecutive epochs. Omit to disable. |
| `loss_goal` | `float` | `None` | No | Training stops immediately when validation loss drops below this value. Omit to disable. |
| `advanced_metrics` | `bool` | `false` | No | If `true`, computes macro-averaged precision, recall, F1, and confusion matrix on the validation set at each epoch. Stored in `history.json`. |
| `plotting` | `dict` | `None` | No | Controls training visualisations. See below. |

**`train.plotting`:**

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | If `true`, generates plots after training completes and saves them to `<save_dir>/visualisation/train/`. |
| `advanced_metrics` | `bool` | `false` | If `true`, also plots macro precision, recall, F1 curves and the final confusion matrix. Requires `train.advanced_metrics: true`. |

>**Early stopping patience and loss goal** are independent — either, both, or neither can be set. If both are set, training stops as soon as either condition is met.

**Checkpointing behaviour:**

- `checkpoint_epoch_{N}.pth` — saved every `checkpoint_interval` epochs. Contains model weights, optimizer state, epoch number, and val loss.
- `best_model.pth` — updated whenever validation loss improves. Same format as interval checkpoints.
- `last_model.pth` — saved at the end of training. Contains model weights only.

All checkpoints are written to `<save_dir>/checkpoints/`. If `save_dir` is not set, no checkpoints are written to disk but the best model state is kept in memory and returned.

### optimizer

Nested under `train.optimizer`.

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | `str` | — | Yes | Optimizer to use. Valid values: `sgd`, `adam`, `adamw`. |
| `lr` | `float` | `0.001` | No | Learning rate. |
| `weight_decay` | `float` | `0.0` (`sgd`, `adam`) / `0.01` (`adamw`) | No | L2 regularisation factor. |
| `momentum` | `float` | `0.9` | No | Momentum factor. **SGD only**, ignored by other optimizers. |
| `scheduler` | `dict` | `None` | No | Learning rate scheduler. Omit the key entirely to train with a fixed learning rate. See [scheduler](#scheduler). |

### scheduler

Nested under `train.optimizer.scheduler`.

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Scheduler to use. Valid values: `cosine`, `step`. |

**`cosine` — CosineAnnealingLR:**

| Key | Type | Default | Description |
|---|---|---|---|
| `T_max` | `int` | — | Number of epochs for one cosine annealing cycle. Typically set to the total number of training epochs. |
| `eta_min` | `float` | `0` | Minimum learning rate at the end of the cycle. |

**`step` — StepLR:**

| Key | Type | Default | Description |
|---|---|---|---|
| `step_size` | `int` | — | Number of epochs between each learning rate decay step. |
| `gamma` | `float` | `0.1` | Multiplicative factor applied to the learning rate at each step. |

---

## test

| Key | Type | Default | Description |
|---|---|---|---|
| `advanced_metrics` | `bool` | `false` | If `true`, computes macro-averaged precision, recall, F1, confusion matrix, and per-class precision, recall, and F1 on the test set. All metrics are saved to `<save_dir>/test_metrics.json`. |
| `plotting` | `dict` | `None` | Controls test visualisations. See below. |

**`test.plotting`:**

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | If `true`, generates plots after evaluation and saves them to `<save_dir>/visualisation/test/`. |