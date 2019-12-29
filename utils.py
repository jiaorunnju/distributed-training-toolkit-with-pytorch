import torch.optim as optim
import torch

# get configurations
from config import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file("settings.yaml")
cfg.freeze()


def get_optimizer(params, cfg):
    name = cfg.OPTIMIZER.NAME
    all_optimizers = sorted(name for name in optim.__dict__
                            if (not name.endswith('Error') and not name.endswith('Exception')
                                and not name.startswith('__') and name[0].isupper()))
    assert name in all_optimizers, "undefined optimizer {0}".format(name)

    oc = {
        'lr': cfg.OPTIMIZER.LR,
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

    if name == 'Adadelta':
        return optim.__dict__[name](params, lr=oc['lr'], rho=oc['rho'], weight_decay=oc['weight_decay'])
    elif name == 'Adagrad':
        return optim.__dict__[name](params, lr=oc['lr'], lr_decay=oc['lr_decay'], weight_decay=oc['weight_decay'])
    elif name == 'Adam':
        return optim.__dict__[name](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'],
                                    amsgrad=oc['amsgrad'])
    elif name == 'AdamW':
        return optim.__dict__[name](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'],
                                    amsgrad=oc['amsgrad'])
    elif name == 'SparseAdam':
        return optim.__dict__[name](params, lr=oc['lr'], betas=oc['betas'])
    elif name == 'Adamax':
        return optim.__dict__[name](params, lr=oc['lr'], betas=oc['betas'], weight_decay=oc['weight_decay'])
    elif name == 'ASGD':
        return optim.__dict__[name](params, lr=oc['lr'], lambd=oc['lambd'], weight_decay=oc['weight_decay'],
                                    alpha=oc['alpha'], t0=oc['t0'])
    elif name == 'LBFGS':
        return optim.__dict__[name](params, lr=oc['lr'], max_iter=oc['max_iter'], history_size=oc['history_size'])
    elif name == 'RMSprop':
        return optim.__dict__[name](params, lr=oc['lr'], alpha=oc['alpha'], weight_decay=oc['weight_decay'],
                                    momentum=oc['momentum'], centered=oc['centered'])
    elif name == 'Rprop':
        return optim.__dict__[name](params, lr=oc['lr'], etas=oc['etas'], step_sizes=oc['step_sizes'])
    elif name == "SGD":
        return optim.__dict__[name](params, lr=oc['lr'], weight_decay=oc['weight_decay'],
                                    momentum=oc['momentum'], nesterov=oc['nesterov'])
    else:
        raise RuntimeError("undefined optimizer in pytorch")


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


if __name__ == '__main__':
    x = [torch.autograd.Variable(torch.Tensor([1]), requires_grad=True), ]
    op = get_optimizer(x, cfg)
    print(op)
