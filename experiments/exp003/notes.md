# Experiment Notes

## Exp003 — ResNet50 scratch with AdamW and Cosine

**Date:** 2026-06-24
**Config:** `experiments/exp003/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | resnet50 |
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
| Train | 0.2668 | 0.9307 |
| Val | 2.4562| 0.3832 |
| Test | 2.4245 | 0.2908 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.253 |
| Macro Recall | 0.2487 |
| Macro F1 | 0.229 |

### Observations

First, training time decreased considerably from VGG16 at ~49 seconds per epoch
to ~25 seconds, which makes sense taking into account that ResNet50 has fewer less
parameters (~25M compared to ~138M).

While the train loss kept decreasing almost linearly, the val loss oscillated at
~2.5 during training with no signal of stabilising. Train accuracy was increasing
almost linearly too but val accuracy decoupled from epoch 3 and although it was
improving it was with a slower pace. 

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy.

In the per-class precision, recall and F1 we can see some classes with 0 score
again. The confusion matrix shows that the model still confuse art deco with
art noveau, they can be similar so probably this wont change in different models.
The model also has classified different styles as deconstructivism and art noveau
which means it has not make a clear representation of what each style is. This 
two type of failures (ambigous pairs and broadly scattered classes) are different,
the first one is a difficulty from the dataset, while the other is a signal of
the model not generalising well. While the second might be corrected, the first
one is harder as it is an intrinsic difficulty of the dataset. 


### Interpretation

While in the training of VGG16 we saw how val loss kept increasing while train
loss was decreasing in the resnet50 we can see that val loss has been oscillating
for almost all the training. Although it started decoupling at epoch 3/4, the
spike in epoch 8 worsened it, which made the val loss to start oscillating with
no clear signal of improving.

Train loss decreased much faster in VGG16 training than in ResNet50, problably
VGG16 model overfit more agressively than ResNet50 and while VGG16 settle fast,
ResNet50 was still very sensitive to which batch it sees in validation. This is
due to ResNet50 having less parameters (~25M) than VGG16 (~138M) so it cannot
memorize the training set completely as the VGG16 did.

### Next

Even if the model has overfitted again, it is not as much as in the VGG16 
architecture, so finetuning with pretrained weights can make a difference and
we might see that the model can start generalising styles without overfitting
to the dataset that much. The rest of the parameters will be the same.

---