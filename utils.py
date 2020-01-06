import torch.optim as optim
import torch


def get_optimizer(params, cfg):
    type = cfg.OPTIMIZER.TYPE
    all_optimizers = sorted(name for name in optim.__dict__
                            if (not name.endswith('Error') and not name.endswith('Exception')
                                and not name.startswith('__') and name[0].isupper()))
    assert type in all_optimizers, "undefined optimizer {0}".format(type)

    oc = {
        'lr': cfg.OPTIMIZER.BASE_LR * cfg.SYSTEM.NUM_GPUS,
        'rho': cfg.OPTIMIZER.RHO,
        'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY,
        'lr_decay': cfg.OPTIMIZER.LR_DECAY,
        'betas': cfg.OPTIMIZER.BETAS,
        'amsgrad': cfg.OPTIMIZER.AMSGRAD,
        'lambd': cfg.OPTIMIZER.LAMBD,
        'alpha': cfg.OPTIMIZER.ALPHA,
        't0': cfg.OPTIMIZER.T0,
        'momentum': cfg.OPTIMIZER.MOMENTUM,
        'centered': cfg.OPTIMIZER.CENTERED,
        'etas': cfg.OPTIMIZER.ETAS,
        'step_sizes': cfg.OPTIMIZER.STEP_SIZES,
        'nesterov': cfg.OPTIMIZER.NESTEROV,
        'max_iter': cfg.OPTIMIZER.MAX_ITER,
        'history_size': cfg.OPTIMIZER.HISTORY_SIZE,
    }

    if type == 'Adadelta':
        return optim.__dict__[type](params, lr=oc['lr'], rho=oc['rho'], weight_decay=oc['weight_decay'])
    elif type == 'Adagrad':
        return optim.__dict__[type](params, lr=oc['lr'], lr_decay=oc['lr_decay'], weight_decay=oc['weight_decay'])
    elif type == 'Adam':
        return optim.__dict__[type](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'],
                                    amsgrad=oc['amsgrad'])
    elif type == 'AdamW':
        return optim.__dict__[type](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'],
                                    amsgrad=oc['amsgrad'])
    elif type == 'SparseAdam':
        return optim.__dict__[type](params, lr=oc['lr'], betas=oc['betas'])
    elif type == 'Adamax':
        return optim.__dict__[type](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'])
    elif type == 'ASGD':
        return optim.__dict__[type](params, lr=oc['lr'], lambd=oc['lambd'], weight_decay=oc['weight_decay'],
                                    alpha=oc['alpha'], t0=oc['t0'])
    elif type == 'LBFGS':
        return optim.__dict__[type](params, lr=oc['lr'], max_iter=oc['max_iter'], history_size=oc['history_size'])
    elif type == 'RMSprop':
        return optim.__dict__[type](params, lr=oc['lr'], alpha=oc['alpha'], weight_decay=oc['weight_decay'],
                                    momentum=oc['momentum'], centered=oc['centered'])
    elif type == 'Rprop':
        return optim.__dict__[type](params, lr=oc['lr'], etas=oc['etas'], step_sizes=oc['step_sizes'])
    elif type == "SGD":
        return optim.__dict__[type](params, lr=oc['lr'], weight_decay=oc['weight_decay'],
                                    momentum=oc['momentum'], nesterov=oc['nesterov'])
    else:
        raise RuntimeError("undefined optimizer {0}".format(type))


def get_scheduler(optimizer, last_epoch, cfg):
    type = cfg.SCHEDULER.TYPE
    step_size = cfg.SCHEDULER.STEP_SIZE
    gamma = cfg.SCHEDULER.GAMMA
    mile_stones = cfg.SCHEDULER.MILE_STONES

    if type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
    elif type == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, gamma, last_epoch)
    elif type == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch)
    elif type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.SCHEDULER.T_MAX, cfg.SCHEDULER.MIN_LR,
                                                         last_epoch)
    else:
        raise RuntimeError("undefined learning rate scheduler {0}".format(type))

    if cfg.SCHEDULER.WARM_UP:
        return WarmupScheduler(optimizer, cfg.SCHEDULER.WARM_UP_EPOCHS, scheduler, last_epoch)
    else:
        return scheduler


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.scheduler = after_scheduler
        self.warmup_epochs = warmup_epochs
        self.finish_warmup = False
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finish_warmup:
                self.finish_warmup = True
            return self.scheduler.get_lr()
        else:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finish_warmup and self.scheduler:
            if epoch is None:
                return self.scheduler.step(None)
            else:
                return self.scheduler.step(epoch - self.warmup_epochs)
        else:
            return super(WarmupScheduler, self).step(epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
