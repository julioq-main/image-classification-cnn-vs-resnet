# Experiment Notes

## Exp008 — ConvNeXt-Tiny finetune with AdamW and Cosine

**Date:** 2026-06-25
**Config:** `experiments/exp008/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | convnext_tiny |
| Pretrained | Yes |
| Epochs run | 14 (max was 50) |
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
| Train | 0.0299 | 0.9884 |
| Val | 0.8740 | 0.8008 |
| Test | 0.6479 | 0.8127 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.8021 |
| Macro Recall | 0.7752 |
| Macro F1 | 0.7808 |

### Observations

The train loss kept decreasing reaching almost 0 with a slight increase at epoch
13. The val loss decreased until epoch 4 at ~0.6, when it started to increase 
slowly up to ~0.9. Train accuracy has same pattern but inversed, incresing until
almost reaching 1 with a slight decrease in epoch 13 and val accuracy increasing
started increasing from ~0.7 until epoch 4 when it stagnated at ~0.8 accuracy. 

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy
as always.

In the per-class precision, we see the same exact behaviour of the previous 
fintuning runs, 0-score classes disappearing while having higher scores overall.

The confusion matrix does not show any strong confusion between some classes, it
has some scattered noise but overall is has learned the features of each class, 
having more difficulty with the underrepresented classes like american foursquare
or palladian style.

### Interpretation

Comparing the accuracy on the test dataset we see that VGG16 finetune had 0.7091,
ResNet had 0.7749 and efficientnet-b0 had 0.7828 while this model has reached
the mark of 0.8 having 0.8127. Macro precision has reached 0.8 too and recall and 
f1 are a bit behind with ~0.78 but still better than the ones of other models.

However the difference is not that significant to say that there is a big
improvement over efficientnet. It has also more parameters ~28.5M compared to 
~5.3M of efficientnet and longer training times ~44 seconds per epoch compared to
~15 seconds.

Taking into consideration the minimum val loss or test loss, we can see that 
convnext (with ~0.6 and ~0.64 respectively) is the lowest across all the finetuned
models, with VGG16 having ~1 and ~1.05, ResNet having ~0.8 and ~0.85 and 
EfficientNet having ~0.8 and ~0.8 respectively, which also supports the claim that
the model has learned to extract better features and indeed performs better,
probably because the pretrained model has learnt more expessive representations.

The model still overfits to the training data as in previous runs.

Overall we see small improvements over each architecture when finetuning, each
one surpassing its predecessor, but not a giant leap in terms of metrics. It is
clear that each time is more difficult to reach higher scores and and surpassing
0.8 is a real milestone, but is far from reaching 0.9.

And while convnext-tiny performs better than efficientnet it is also more
memory heavy and takes more time to train and run. So depending on the deployment
constraints one model can fit better than other. There is no one-for-all solution.

This is just one-run experiment though and to draw clear conclusion more runs
should be made to have a good statistical result. 

### Next

Now it is time to see if augmentation can really make a difference and make the
model to overfit less to the training data or that it can transfere more of the
learning. We will use this last model as it has the best results finetuned and
the rest of parameters will remain the same.

---