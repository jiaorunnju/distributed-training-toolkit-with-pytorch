import os
import random
import warnings
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from config import get_cfg_defaults
import tasks
from utils import AverageMeter, ProgressMeter, get_optimizer, get_scheduler

# list all defined tasks
all_tasks = sorted(name for name in tasks.__dict__
                   if name[0].isupper() and not name.startswith("__"))

# get configurations
cfg = get_cfg_defaults()
cfg.merge_from_file("settings.yaml")
cfg.freeze()

# get the train task
assert cfg.TRAIN.TASK in all_tasks, "undefined task {0}".format(cfg.TRAIN.TASK)
task = tasks.__dict__[cfg.TRAIN.TASK]()

# whether use fp16
if cfg.SYSTEM.FP16 is True:
    from apex import amp

# keep the best model so far
best_metric = 0


def main():
    if cfg.TRAIN.SEED is not -1:
        random.seed(cfg.TRAIN.SEED)
        torch.manual_seed(cfg.TRAIN.SEED)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if cfg.SYSTEM.NUM_GPUS > 1:
        # use multiprocessing
        mp.spawn(main_worker, nprocs=cfg.SYSTEM.NUM_GPUS, args=())
    else:
        # single-card training
        main_worker(None)


def main_worker(gpu):
    global best_metric

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    # initialize local variables according to cfg
    distributed = True if cfg.SYSTEM.NUM_GPUS > 1 else False
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.SYSTEM.NUM_WORKERS
    start_epoch = cfg.TRAIN.START_EPOCH

    # initialize process group
    if distributed:
        dist.init_process_group(backend=cfg.SYSTEM.BACKEND, init_method=cfg.SYSTEM.DIST_URL,
                                world_size=cfg.SYSTEM.NUM_GPUS, rank=gpu)

    # initialize model
    print("=> creating model...")
    model = task.get_model()
    # loss and optimizer
    criterion = task.get_criterion().cuda(gpu)
    optimizer = get_optimizer(model.parameters(), cfg)

    if gpu is not None:
        # distributed training
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        if cfg.SYSTEM.FP16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.SYSTEM.OP_LEVEL)
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.SYSTEM.NUM_GPUS)
        num_workers = int((cfg.SYSTEM.NUM_WORKERS + cfg.SYSTEM.NUM_GPUS - 1) / cfg.SYSTEM.NUM_GPUS)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        # non-distributed
        model = torch.nn.DataParallel(model).cuda()
        if cfg.SYSTEM.FP16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.SYSTEM.OP_LEVEL)

    if len(cfg.TRAIN.RESUME_FROM) > 0:
        # resume from checkpoint
        if os.path.isfile(cfg.TRAIN.RESUME_FROM):
            print("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME_FROM))
            if gpu is None:
                checkpoint = torch.load(cfg.TRAIN.RESUME_FROM)
            else:
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(cfg.TRAIN.RESUME_FROM, map_location=loc)
            start_epoch = checkpoint["epoch"]
            best_metric = checkpoint["best_metric"]
            best_metric = best_metric.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # load amp state_dict
            if cfg.SYSTEM.FP16:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.TRAIN.RESUME_FROM, checkpoint['epoch']))
        else:
            warnings.warn("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME_FROM))

    # define scheduler
    scheduler = get_scheduler(optimizer, last_epoch=start_epoch-1, cfg=cfg)

    # may accelerate the computation
    cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK

    # get the dataset and build iterator
    train_dataset = task.get_train_dataset(cfg.TRAIN.TRAIN_DATA)
    valid_dataset = task.get_valid_dataset(cfg.TRAIN.VALID_DATA)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed \
        else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=cfg.SYSTEM.PIN_MEMORY, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=cfg.SYSTEM.PIN_MEMORY
    )

    # training steps
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if distributed:
            train_sampler.set_epoch(epoch)

        # print learning rate
        print("Time: {0}".format(time.asctime(time.localtime(time.time()))),
              " Learning Rate: {0}".format(scheduler.get_lr()[0]))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, gpu)

        # evaluate on validation set
        metric1 = validate(val_loader, model, criterion, gpu)

        # remember best metric and save checkpoint
        is_best, best_metric = task.update_metric(best_metric, metric1)

        # save checkpoint
        if not os.path.isdir(cfg.TRAIN.CHECKPT_PATH):
            os.mkdir(cfg.TRAIN.CHECKPT_PATH)
        ckpt_filename = os.path.join(cfg.TRAIN.CHECKPT_PATH, "{0}_epoch_{1}.pt".format(cfg.TRAIN.MODEL_NAME, epoch))
        if not distributed or gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'optimizer': optimizer.state_dict(),
                'amp': None if not cfg.SYSTEM.FP16 else amp.state_dict()
            }, is_best, filename=ckpt_filename)

        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    metric_dict = task.get_metric()
    metric_list = [AverageMeter(i, ':6.2f') for i in metric_dict['name']]
    metric = metric_dict["metric"]

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses] + metric_list,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (source, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            source = source.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(source)
        loss = criterion(output, target)

        # measure accuracy and record loss
        m_out = metric(output, target)
        losses.update(loss.item(), source.size(0))
        for m, o in zip(metric_list, m_out):
            m.update(o[0], source.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if cfg.SYSTEM.FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.TRAIN.PRINT_FREQ == 0:
            progress.display(i)


def validate(val_loader, model, criterion, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    metric_dict = task.get_metric()
    metric_list = [AverageMeter(i, ':6.2f') for i in metric_dict['name']]
    metric = metric_dict["metric"]

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses] + metric_list,
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (source, target) in enumerate(val_loader):
            if gpu is not None:
                source = source.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(source)
            loss = criterion(output, target)

            # measure accuracy and record loss
            m_out = metric(output, target)
            losses.update(loss.item(), source.size(0))
            for m, o in zip(metric_list, m_out):
                m.update(o[0], source.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(i)

        if gpu == 0:
            summary = [name + ": " + str(round(float(val.avg), 3)) for name, val in zip(metric_dict['name'], metric_list)]
            print(*summary)

    return metric_list[0].avg


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(cfg.TRAIN.CHECKPT_PATH, 'model_best.pt'))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg.OPTIMIZER.LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
