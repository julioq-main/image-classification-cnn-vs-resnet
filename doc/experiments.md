Experiments

Test vgg16 to see batch size can handle

## exp001
No augmentaiton
VGG16
50/100? epochs
AdamW
cosine

## exp002
No augmentaiton
VGG16 finetune
50 epochs
AdamW
cosine

## exp003
No augmentaiton
resnet from scratch
50 epochs
AdamW
cosine scheduler

## exp004
No augmentaiton
resnet finetune
50 epochs
AdamW
cosine scheduler

## exp005
No augmentaiton
EfficientNet from scratch
50 epochs
AdamW
cosine scheduler

## exp006
No augmentaiton
EfficientNet finetune
50 epochs
AdamW
cosine scheduler

## exp007
no augmentaiton
ConvNext from scratch
50 epochs
AdamW
cosine scheduler


## exp008
no augmentaiton
ConvNext finetune
50 epochs
AdamW
cosine scheduler


## exp009
augmentaiton
best finetune
50 epochs
AdamW
cosine scheduler



## exp010
no augmentaiton
resnet finetuned
50 epochs
SGD + momentum
cosine scheduler


## exp011
no augmentaiton
resnet finetuned
50 epochs
AdamW
No scheduler