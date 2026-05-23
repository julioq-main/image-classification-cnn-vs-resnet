# Experiment Notes

## Exp002 — VGG16 finetuned with AdamW and Cosine

**Date:** 2026-06-23  
**Config:** `experiments/exp002/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | vgg16 |
| Pretrained | Yes |
| Epochs run | 15 (max was 50) |
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
| Train | 0.0179 | 0.9900 |
| Val | 1.7735 | 0.7173 |
| Test | 1.0562 | 0.7091 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.6969 |
| Macro Recall | 0.6651 |
| Macro F1 | 0.6673 |

### Observations

The training curves show that while train loss has declined to almost reach 0
the val loss has been slowly growing. With the accuracy we see the train  one 
reaching almost 1 while the val one remains at ~0.7 from the begining.

Macro-averaged precision, recall and F1 remains at ~0.7 with small increases over
epochs too.

On the test part, per-class precision, recall and F1 are also higher with no one
having a 0 score, but still high differences between some classes. The confusion
matrix does not offer any pattern of which classes can be confused with other,
except art-deco and art-nouveau.

### Interpretation

While the pretrained weights have improved the base loss and accuracy by a lot
(it starts with ~1.25 val loss and ~0.6 val accuracy) we still see that the
model overfits to the dataset, having train loss converging to 0 while val loss
increases.

Macro-averaged precision, recall and F1 been at ~0.7 also tell us the help of
pretraining on the base accuracy.

On the test metrics, we see around same accuracy as in val and more or less the 
same minimum val loss (~1) achieved at epoch 5.

The pretraining has improved all the per-class metrics, helping the 
underrepresented classes to not get 0 scores. It also has made the differences
between precision, recall and f1 scores in some classes less notable, having
almost same scores in many classes.

### Next

Even if the model is pretrained on the imagenet dataset, the size of this 
dataset (~3800) is too small for a model this big (~138M parameters). It 
performs better than the from-scratch one but only because of the pretrained
weights as the val loss has not decreased much from the first epoch.

It is clear that, while this architecture might benefit from bigger datasets
it is not suitable for a task with a smaller size, even augmentations might not
make that much of a difference. Therefore, in the next experiment we are going 
to train the next model on the list, the ResNet50, to see if having residual
layers and a smaller parameter size (~28M) can make an impact on smaller sized 
tasks or if it will also overfit.

The rest of parameters will be maintained the same to isolate the effect of
changing the model architecture.

---