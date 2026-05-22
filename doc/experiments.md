Experiments

Test vgg16 to see batch size can handle

## exp001
No augmentaiton
Base VGG16
50 epochs
sgd
no scheduler

## exp002
No augmentaiton
Base VGG16 from scratch
50 epochs
sgd
cosine scheduler

## exp003
No augmentaiton
Base VGG16 from scratch
50 epochs
adam/adamw

## exp004
no augmentaiton
Base VGG16 finetune
50 epochs
sgd
cosine scheduler

## exp005
No augmentaiton
Base VGG16 finetune
50 epochs
adam/adamw

## exp004
all augmentaiton
Base VGG16 finetune
50 epochs
best sgd/adam/adamw
cosine scheduler?