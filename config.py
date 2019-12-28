from yacs.config import CfgNode as CN
import os


_C = CN()

# configs for system
_C.SYSTEM = CN()
# number of gpus used
_C.SYSTEM.NUM_GPUS = 8
# number of workers to load data
_C.SYSTEM.NUM_WORKERS = 4
# whether use pin_memory
_C.SYSTEM.PIN_MEMORY = True
# backend for ddp
_C.SYSTEM.BACKEND = 'nccl'
# init url for ddp
_C.SYSTEM.DIST_URL = 'tcp://127.0.0.1:23456'
# FP-16 support
_C.SYSTEM.FP16 = False

# configs for training
_C.TRAIN = CN()
# task name
_C.TRAIN.TASK = 'ImageClassifyTask'
# learning rate
_C.TRAIN.LR = 0.1
# momentum
_C.TRAIN.MOMENTUM = 0.9
# weight decay
_C.TRAIN.WEIGHT_DECAY = 1e-4
# data path
_C.TRAIN.DATA = "data"
_C.TRAIN.TRAIN_DATA = os.path.join(_C.TRAIN.DATA, "train")
_C.TRAIN.VALID_DATA = os.path.join(_C.TRAIN.DATA, "valid")
_C.TRAIN.TEST_DATA = os.path.join(_C.TRAIN.DATA, "test")
# checkpoint path
_C.TRAIN.CHECKPT_PATH = "checkpoint"
# model name
_C.TRAIN.MODEL_NAME = "model"
# random seed
_C.TRAIN.SEED = -1
# resume training
_C.TRAIN.RESUME_FROM = ""
# batch size
_C.TRAIN.BATCH_SIZE = 512
# start epoch
_C.TRAIN.START_EPOCH = 0
# total epochs
_C.TRAIN.EPOCHS = 100
# benchmark
_C.TRAIN.CUDNN_BENCHMARK = False
# print freq
_C.TRAIN.PRINT_FREQ = 10


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
