# Distributed Training in Pytorch

This is a code template for distributed training in pytorch. It
supports single-card or multi-card training on a single machine.
It is based on DistributedDataParallel and Apex, thus it can
perform better than DataParallel.

## Requirements

### pytorch with gpu-support
Install instructions can be found [here](https://pytorch.org/).

### Apex
If you want to training with FP16 support, you should install
[Apex](https://github.com/NVIDIA/apex) first. A simple tutorial of
mixed-precision training with Apex can be found 
[here](http://on-demand.gputechconf.com/gtc-cn/2018/pdf/CH8302.pdf).

## Usage
To use this code template, you need to define a task. The abstract
class of a task is in **tasks/train_task.py**. You need to specify
the model, dataset, loss and metrics for evaluation. After this, the 
training progress is handled by this code template.

### An example for Imagenet training
See **tasks/image_classify_task.py** for details.

## Performance
On our server with 4 Nvidia V100 gpus, we can train a resnet-50
to achieve 76.57% top-1 accuracy using standard data augments(flip, crop, normalize) within 16 hours.

Hyperparameters are:
- 100 epochs
- base learning rate 0.2
- linear scaling of learning rate to 0.2*4=0.8
- SGD with Nesterov momentum 0.9
- weight-decay 1e-4
- batch size 2048
- Apex optimization level *O1*
- CosineAnnealing learning rate scheduler
- warm up for 5 epochs with linear learning rates

Some reminds
- set enough workers for data loader
- Exponential scheduler may decay too fast
- tips on large-batch training: [Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)
- warm up is very important for large batch training, since
the training may diverge(batch-size > 2k in our case) if using a large learning rate at the start.
