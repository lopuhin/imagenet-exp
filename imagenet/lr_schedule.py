"""
Different kinds of learning rate schedules.
"""


def imagenet_schedule(optimizer, epoch, args):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    _set_lr(optimizer, lr)


def linear_schedule(optimizer, epoch, args):
    """ Same as imagenet_schedule but with a small drop each epoch.
    """
    lr = args.lr * (0.1 ** (epoch / 30))
    print('lr', lr)
    _set_lr(optimizer, lr)


def _set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
