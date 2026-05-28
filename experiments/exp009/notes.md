# Experiment Notes

## Exp009 — ConvNeXt-Tiny finetune with AdamW and Cosine and augmentation

**Date:** 2026-06-27
**Config:** `experiments/exp009/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | convnext_tiny |
| Pretrained | Yes |
| Epochs run | 20 (max was 50) |
| Stopped by | Early stopping |
| Optimizer | AdamW |
| LR | 0.0001 |
| Weight decay | 0.001 |
| Scheduler | Cosine (T_max=50) |
| Augmentation | Yes |
| Batch size | 32 |

### Results

| Split | Loss | Accuracy |
|---|---|---|
| Train | 0.1908 | 0.9388 |
| Val | 0.5711 | 0.8629 |
| Test | 0.6175 | 0.8266 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.8085 |
| Macro Recall | 0.7929 |
| Macro F1 | 0.7978 |

### Observations

This time the train loss did not reach values near 0 but ~0.2, the variance of
images provided by the augmentations did not let the model memorise the dataset.
The val loss decreased reaching ~0.5 it decoupled at epoch 7 this time and 
stabilised at that value from then. Train accuracy kept increasing reaching ~0.9
while val accuracy decoupled at epoch 7 more or less stagnated from then. 

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy
as always.

In the per-class precision, we see the same exact behaviour of the previous 
fintuning runs, 0-score classes disappearing while having higher scores overall.

The confusion matrix does not show any strong confusion between some classes, it
has some scattered noise but overall is has learned the features of each class, 
having more difficulty with the underrepresented classes like american foursquare.

### Interpretation

We can see how augmentation has increased the val accuracy and overall performance
of the model.

The main difference is how the curves of val and train stayed closer than in 
previous runs showing that augmentation indeed makes the model overfits less and
this lead to better performing models as the val metrics relates to the train 
metrics during more epochs.

More detailed interpretation will be provided in experiments.md with all the data.
---