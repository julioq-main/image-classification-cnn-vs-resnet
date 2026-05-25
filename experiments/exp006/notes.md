# Experiment Notes

## Exp006 — EfficientNet-B0 finetune with AdamW and Cosine

**Date:** 2026-06-25
**Config:** `experiments/exp006/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | efficientnet_b0 |
| Pretrained | Yes |
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
| Train | 0.0403 | 0.9853 |
| Val | 0.7803 | 0.7901 |
| Test | 0.8001 | 0.7828 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.7572 |
| Macro Recall | 0.7475 |
| Macro F1 | 0.7467 |

### Observations

As always, the train loss kept decreasing reaching almost 0, while val loss
decoupled at epoch 4 and decreased a bit more until stagnating from epoch 7 at
~0.8. The accuracy also shows a similar pattern but inversed, the train accuracy
almost reaching 1 while the val accuracy decouple at epoch 4 and stagnating from
epoch 7 onwards at ~0.8.

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy
as always, althugh precision score is a bit higher than others at the beginning
it is probably noise and not a tangible difference.

In the per-class precision, we see the same exact behaviour of the previous 
fintuning runs, 0 score classes disappearing while having higher scores overall.

The confusion matrix show some confusion  between art deco and art noveau and
some scattered noise but overall is has learned the features of each class, 
having more difficulty with the underrepresented classes like american foursquare
or palladian style.

### Interpretation

Comparing the accuracy on the test dataset we see that VGG16 finetune had 0.7091
and ResNet had 0.7749 while this model has 0.7828. Macro precision, recall and 
f1 are also similar to the ones of resnet. So we cannot say that is there any
real advantage for using efficientnet over resnet if we compare the accuracy, it
is true though that efficientnet is faster to train and inference on and having
less number of parameters (~5.3M compared to 25M of resnet) makes it easier to
train on less capable hardware. Both are better than VGG16 in this case.

Also val loss was decreasing for more epochs, while in exp002 and exp004 it
nearly achieved the minimum at epoch 2. After that, the loss almost stabilised
rather than increasing.

While the model stills overfit and almost memorise the training dataset it
manages to get ~0.8 accuracy on the test set using the checkpoint of the epoch 7.
So in just a few minutes of training you can get a model that gets that accuracy
using the pretrained weights. 

### Next

On the next experiment, we will see a new architecture, Convnext to see if this
new design can achieve a significal improvement over resnet or efficientnet,
both from scratch and while finuting. After that, we will see if augmentation
can make any difference on the best performing model.

---