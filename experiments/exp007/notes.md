# Experiment Notes

## Exp007 — ConvNeXt-Tiny scratch with AdamW and Cosine

**Date:** 2026-06-25
**Config:** `experiments/exp007/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | convnext_tiny |
| Pretrained | No |
| Epochs run | 17 (max was 50) |
| Stopped by | Early stopping |
| Optimizer | AdamW |
| LR | 0.0001 |
| Weight decay | 0.001 |
| Scheduler | Cosine (T_max=50) |
| Augmentation | No |
| Batch size | 32 |

### Results

| Split | Loss | Accuracy |
|---|---|---|
| Train | 0.1637 | 0.9780 |
| Val | 3.6523 | 0.2612 |
| Test | 2.6564 | 0.2171 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.1486 |
| Macro Recall | 0.1641 |
| Macro F1 | 0.1454 |

### Observations

The train loss has decreased almost linearly reaching ~0.16, its behaviour is
similar to the one seen in exp002 using resnet from scratch. Val loss, on the
other hand, decoupled at epoch 4 and stayed flat until epoch 8 when it started
increasing smoothly. The train accuracy increased almost linearly too, reaching
almost 1, while val accuracy decoupled at epoch 4 and increased a little bit
more until epoch 8 where it stayed more or less flat.

Macro-averaged precision, recall and F1 show the same pattern of val accuracy but
with a traslation of ~-0.1 on the Y axis.

In the per-class precision, recall and F1 there are classes with 0 score, more 
than in previous runs with the other architectures, the scores are low except
ancient egyptcion architecture wich has a really high score (~0.8), in the previous
runs it has been high too, so it is probably because of the distinct features
the ancient egyptian architecture has.

The confusion matrix shows a lot of guesses scattered, it confuses art deco and
art noveaus as previous models but also with baroque. However it seems it has used
this three classes as well as deconstructivism and novelty architecture as
garbage collector as there are a lot of guesses.

### Interpretation

In general it has performed worse (~0.21 in accuracy) than the previous models
(~0.4 in VGG16, ~0.29 in resnet and ~0.33 in efficientnet). Also the training
time is ~44 seconds which is similar to the one of VGG16.

As the architecture of VGG16 is simpler (although it has more parameters than the
others) the loss lanscape is easier to train from scratch to find better minima.
So far, the best performing model for this task has been VGG16. When finetuning
the picture is different. Having been pretrained with larger dataset, the initial
weights reside near good minima, makin the task easier.

Also, convnext tiny has ~28.5M parameters wich makes the model overfit more than
in efficientnet with ~5.3M. 

### Next

While this shows that using more complex architectures do indeed makes training
a model harder when doing from scrath in such small dataset, finetuning with
pretrained weights has enhanced its capabilities, we will see how this can affect
this model and if it can benefit from pretrained weights enough to beat the other
models.

---