# Experiments

## 1. Introduction

These experiments, as well as serving as a demonstration of the pipeline, are
intended to explore how the architecture designs for Convolutional Neural 
Networks have evolved throughout the years and how it affects the performance for
a small-sized classification task. Through a small selection of different
architectures we compare the evolution of performance across them and explain 
the causes of the differences in the performance.

The architectures that will be studied are:

- VGG16 (2014): demonstrated that depth using small 3×3 convolutions consistently
outperforms shallower networks with larger filters.
- ResNet50 (2015): introduced residual connections, solving the vanishing gradient
problem and enabling much deeper networks.
- EfficientNet-B0 (2019): proposed compound scaling to balance depth, width, 
and resolution simultaneously.
- ConvNeXt-Tiny (2022): modernized the classic ConvNet design by incorporating 
ideas from Vision Transformers.

We will also study how finetuning affect performance and on which architectures
this is more noticeable. Finally, there is one experiment showing the effect 
that dataset augmentation has on the model performance.

The dataset used is the [Architecture Dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset)
from Kaggle: 25 architectural style classes with approximately 4,800 images in
total. It is imbalanced, i.e., some classes have significantly fewer examples than 
others.


**Fixed setup across all experiments:**

| Field | Value |
|---|---|
| Optimizer | AdamW |
| LR | 0.0001 |
| Weight decay | 0.001 |
| Scheduler | Cosine (T_max=50) |
| Augmentation | No (except exp009) |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 |

The optimizer AdamW with the scheduler CosineAnnealing was chosen as it is usually 
the most reliable default across all architectures. Every hiperparameter was 
fixed for all experiments to isolate the effect of them on the performance.

The experiments were seeded in order to be able to reproduce them with as much
fidelity as possible.

---

## 2. Scratch Training

In this section all four models were trained from scratch. That means that the
weights started from a random position, so the model had no prior knowledge that
could help with the classification task. This makes the task more difficult for
the models wich is worsened by the small size of the dataset they are trained on.


### Results

| Model | Params | Train Loss | Val Loss | Test Loss | Test Acc | Macro Precision | Macro Recall | Macro F1 | Training time/epoch |
|---|---|---|---|---|---|---|---|---|---|
| VGG16 | ~138M | 0.0313 | 4.0714 | 2.0321 | 0.4063 | 0.3792 | 0.3546 | 0.3453 | ~49s |
| ResNet50 | ~25M | 0.2668 | 2.4562 | 2.4245 | 0.2908 | 0.2530 | 0.2487 | 0.2290 | ~25s |
| EfficientNet-B0 | ~5.3M | 0.5219 | 2.9414 | 2.4249 | 0.3326 | 0.2577 | 0.2565 | 0.2407 | ~15s |
| ConvNeXt-Tiny | ~28.5M | 0.1637 | 3.6523 | 2.6564 | 0.2171 | 0.1486 | 0.1641 | 0.1454 | ~44s |

### Key findings

The difference between the train loss and val loss and the graphs themselves 
reveal that all models overfit significantly. The most notable is VGG16, whith a
train loss of 0.0313 over a val loss of 4.0714 at the last epoch. This is due to
the small size of the dataset: these models were designed for bigger datasets, 
such as ImageNet, where they had to recognise a wide variety of images (one of a
cat, of a human, of a building or of a car, for example). They have, therefore 
millions of parameters in order to be able to extract different patterns for many
things. When training these models with our small dataset, and after seing the 
same images over and over again for each epoch, they start to memorize the images
rather than extracting the features that make them able to classify the images.


<table>
  <tr>
    <td align="center">
      <img src="res/training_curves_001.png" alt="VGG16 training curves" width="100%"/>
      <br/>VGG16 training curves
    </td>
    <td align="center">
      <img src="res/training_curves_003.png" alt="ResNet training curves" width="100%"/>
      <br/>ResNet training curves
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="res/training_curves_005.png" alt="EfficientNet-B0 training curves" width="100%"/>
      <br/>EfficientNet-B0 training curves
    </td>
    <td align="center">
      <img src="res/training_curves_007.png" alt="ConvNeXt-Tiny training curves" width="100%"/>
      <br/>ConvNeXt-Tiny training curves
    </td>
  </tr>
</table>

As it can be seen in the graphs, VGG16's val loss started growing after a few
epochs while train loss kept decreasing, this happened with EfficientNet and 
ConvNeXt too. However, ResNet had a different behaviour, after a few epochs the
val loss started oscillating showing a noisier graph. The reason for this 
phenomenon is likely related to the architecture design, though the exact cause
is uncertain.

It is interesting to see that the latest model, ConvNeXt, is the one with the 
worst performance, while the one with the best performance is the first one, VGG16.
First, let's clarify that for small classification tasks like this one, usually 
smaller models are better, as they don't need as many parameters or depth to
be able to extract a small amount of features in order to classify the images
correctly. However, VGG16 is the model with more parameters but it is still the
best one in terms of accuracy. The reason for this is that as new architectures
were developed, they added more complexity on the design that made the optimization
landscape harder to find good local optima. So, when training a VGG16 model, it
is able to find a better local optimum faster than other architectures. Even though
on other architectures the global minimum can be much better than in VGG16, they
can get stuck on worse local minima that makes the performance worse. ConvNeXt,
for example, was designed around training recipes with strong augmentation and
longer schedules in order to overcome that.

One more thing to mention is that, even though VGG16 is the best of all four, 
EfficientNet outperforms ResNet and ConvNeXt, while also being the one with fewer
parameters (~5.3M) and just having a training duration of ~15 seconds per epoch.

---

Next section still to be finished