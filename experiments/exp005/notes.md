# Experiment Notes

## Exp005 — EfficientNet-B0 scratch with AdamW and Cosine

**Date:** 2026-06-25
**Config:** `experiments/exp005/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | efficientnet_b0 |
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
| Train | 0.5219 | 0.8509 |
| Val | 2.9414 | 0.3383 |
| Test | 2.4249 | 0.3326 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.2577 |
| Macro Recall | 0.2565 |
| Macro F1 | 0.2407 |

### Observations

While train loss kept decreasing reaching 0.5, val loss decouple at epoch 5 and 
started increasing since epoch 7. Train accuracy was increasing reaching more 
tahn 0.8 but val accuracy decoupled at epoch 5 and stayed almost flat since then
at ~0.3

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy.

In the per-class precision, recall and F1 there are classes with 0 score, again
this usually are the underrepresented classes.

The confusion matrix show a similar pattern to the one of exp003, there are
scattered guesses, but it seems the model is labelling a lot of images from 
different classes as art noveau and queen anne architecture. These classes have
become like a garbage collector, when the model is not certain of the class it 
defaults to this classes.


### Interpretation

Comparing the accuracy on the test dataset we see that VGG16 scratch had 0.4063
and ResNet had 0.2908 while this model has 0.3326, macro precision, recall and f1
are similar to the ones of resnet though. It is an improvement over ResNet,
however it is worse than VGG16. The sequential architecture might be the reason
to that, as it is easier to train from random initilisations. We can still see 
how, even with ~5.3M parameters the model overfits to the train dataset. It seems
that all models will overfit and more data or augmentation is needed.

Another thing to consider is training time, that for EfficientNet has been ~15
seconds. This mark a clear improvement over both VGG16 with ~49 seconds and
resnet ~25. 

### Next

While a comparison with VGG16 shows some trade-offs (better accuracy or faster 
and less memory heavy) it is somewhat an improvement compared to ResNet, but the
margin is not large enough to make any conclusions. At least, in this scenario.
We will see how having pretrained weights can affet this comparison and if it can
make the margin bigger and even beat VGG16.
---