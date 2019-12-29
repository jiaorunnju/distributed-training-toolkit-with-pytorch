from yacs.config import CfgNode as CN
import os

_C = CN()

'''
configs for system and pytorch
'''
# configs for system
_C.SYSTEM = CN()

# number of gpus used
_C.SYSTEM.NUM_GPUS = 1

# number of workers to load data
_C.SYSTEM.NUM_WORKERS = 6

# whether use pin_memory
_C.SYSTEM.PIN_MEMORY = True

# backend for ddp
_C.SYSTEM.BACKEND = 'nccl'

# init url for ddp
_C.SYSTEM.DIST_URL = 'tcp://127.0.0.1:23456'

# FP-16 support
_C.SYSTEM.FP16 = False

# Apex optimization level
_C.SYSTEM.OP_LEVEL = 'o1'

'''
configs for training
'''
# configs for training
_C.TRAIN = CN()

# task name
_C.TRAIN.TASK = 'ImageClassifyTask'

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

'''
configs for optimizer
'''
_C.OPTIMIZER = CN()

# optimizer
_C.OPTIMIZER.NAME = 'SGD'

# learning rate
_C.OPTIMIZER.LR = 0.01

# momentum in RMSprop, SGD,
_C.OPTIMIZER.MOMENTUM = 0.0

# weight decay
_C.OPTIMIZER.WEIGHT_DECAY = 0.0

# rho in adadelta
_C.OPTIMIZER.RHO = 0.9

# lr_decay in adagrad
_C.OPTIMIZER.LR_DECAY = 0

# betas in adam, adamW, sparseadam, adamax
_C.OPTIMIZER.BETAS = (0.9, 0.999)

# amsgrad in adam, adamW, sparseadam
_C.OPTIMIZER.AMSGRAD = False

# lambd in ASGD
_C.OPTIMIZER.LAMBD = 0.0001

# alpha in ASGD, RMSprop
_C.OPTIMIZER.ALPHA = 0.75

# t0 in ASGD
_C.OPTIMIZER.T0 = 1000000.0

# centered in RMSprop
_C.OPTIMIZER.CENTERED = False

# etas in Rprop
_C.OPTIMIZER.ETAS = (0.5, 1.2)

# step_size in Rprop
_C.OPTIMIZER.STEP_SIZES = (1e-06, 50)

# nesterov in SGD
_C.OPTIMIZER.NESTEROV = False

# max_iter in LBFGS
_C.OPTIMIZER.MAX_ITER = 20

# history_size in LBFGS
_C.OPTIMIZER.HISTORY_SIZE = 100

'''
configs for lr scheduler
'''
_C.SCHEDULER = CN()

# configs for ReduceLROnPlateau
_C.SCHEDULER.VERBOSE = False
_C.SCHEDULER.MODE = 'min'
_C.SCHEDULER.FACTOR = 0.1
_C.SCHEDULER.PATIENCE = 10
_C.SCHEDULER.THRESHOLD = 1e-4
_C.SCHEDULER.MIN_LR = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
