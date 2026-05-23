# Experiment Notes

## Exp001 — VGG16 scratch with AdamW and Cosine

**Date:** 2026-06-23  
**Config:** `experiments/exp001/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | vgg16 |
| Pretrained | No |
| Epochs run | 18 (max was 50) |
| Stopped by | Early stopping |
| Optimizer | AdamW |
| LR | 0.0001 |
| Weight decay | 0.001 |
| Scheduler | Cosine (T_max=50) |
| Augmentation | No |
| Batch size | 32 |

First run was with 0.001 lr and 0.01 weight decay but the model was not learning
anything, so I updated to 0.0001 and 0.001 respectively on the second run.

### Results

| Split | Loss | Accuracy |
|---|---|---|
| Train | 0.0313 | 0.9892 |
| Val | 4.0714 | 0.4282 |
| Test | 2.0321 | 0.4063 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.3792 |
| Macro Recall | 0.3546 |
| Macro F1 | 0.3453 |

### Observations

When looking at the training curves we can see that until epoch 5 both train and
val loss as well as train and val accuracy were on par. But after that, train
loss kept decreasing while val loss started growing going up to ~4. In accuracy,
we can see that while train accuracy kept going closer to 1, the val accuracy
stagnated at ~0.4. 

Macro-averaged precision, recall and F1 were more or less growing but plateauing
to ~0.4 too.

On the test part, some classes got 0 score on per-class precision, recall and F1
while other got almost 1 score. The confusion matrix does not offer any pattern
of which classes can be confused with other, except art-deco and art-nouveau.

### Interpretation

The most relevant data are the training curves, which show clear overfitting of
the model to the training dataset. It probably started overfitting at epoch 5, 
when train and val metrics decoupled. This is certainly because of the small 
dataset (~3800 training images) compared to the amount of parameters of VGG16 
(~138M). Other models may not suffer this problem of overfitting as much and 
augmentation should probably help to this problem to a lesser degree too.

On the test metrics, it is interesting that both loss and accuracy are lower to
val metrics. It is less accurate but has lower loss. The test loss is around the
same that the val loss from epochs 5-8 where the last checkpoint was made, as 
later val loss grew. This is the reason of the mismatch between last val loss 
and test loss. The accuracy are more or less similar, which makes sense taking 
into account that accuracy stagnated to ~0.4 on the val set.

As for some classes getting score of 0, this is probably because of the small 
size of examples of some classes, augmentation should improve that too.


### Next

Clearly, this model with the amount of parameter is not suitable for a task with
this small size dataset. Next experiment will be with same model and parameters
but finetuned to see whether being pretrained with larger dataset can make the
model better for this task.

As 10 epoch patience is enough to see whether the training is plateauing, it will
not be changed.

---