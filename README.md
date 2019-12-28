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
To be done
