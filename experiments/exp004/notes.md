# Experiment Notes

## Exp004 — ResNet50 finetune with AdamW and Cosine

**Date:** 2026-06-24
**Config:** `experiments/exp004/config.yaml`

### Setup

| Field | Value |
|---|---|
| Model | resnet50 |
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
| Train | 0.0387 | 0.9843 |
| Val | 1.1529 | 0.7194 |
| Test | 0.8567 | 0.7749 |

Advanced metrics (test set):

| Metric | Value |
|---|---|
| Macro Precision | 0.7527 |
| Macro Recall | 0.7377 |
| Macro F1 | 0.7348 |

### Observations

Train loss reached almost 0 at the end of the run but val loss started at ~1.25
went down to ~0.8 in  epoch 2 and stayed more or les the same with small 
increases until epoch 10 when it started going up again up to ~1.15. We can
see the same behaviour but inverted in the accuracy. Train accuracy reaching
almos 1 while val accuracy staying relatively the same at ~0.8.

Macro-averaged precision, recall and F1 show the same behaviour of val accuracy.

In the per-class precision, recall and F1 we can see theres no classes with 0 score
again and all classes has higher scores, which means that the pretrained
weights have helped the model to get a better understanding of the architectural
styles.

Interestingly, the confusion matrix only shows some scattered bad guesses, but
the confusion between art deco and art noveau has disappeared almost completely,
which means the model has learned to differentiate between the two classes.
The scattered failures corresponds mainly with the underrepresented classes, 
which means the model has not seen enough examples to correctly categorise them.


### Interpretation

We can see a similar picture of the exp001 and exp002, the pretrained has made
the base loss and accuracy much better, but the model still overfits. In fact,
the good results of the classification comes almost exclusively from the 
pretrained weights and not the training, as it has stayed almost the same except
for the first epoch. 

While VGG16 test accuracy was ~0.7091, ResNet50 had a test accuracy of ~0.7749, 
which is better, however the gap is small given the architectural difference.
Macro-averaged metrics also improved more or less the same. The biggest advantage
in this task is that training ResNet50 is faster than VGG16. 

### Next

It is clear that the model still overfits and that having pretrained weights 
has not solved it. Next experiment will have the same parameters as the exp003
but with a new architecture, EfficientNet-B0. It has even less parameters
(~5.3M) which should make overfitting harder as it has less parameter to 
memorize the training dataset. We will need to see if the compoind scaling approach
actually gives better pretrained features for finetuning too.

---