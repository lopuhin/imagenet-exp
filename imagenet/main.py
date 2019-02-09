#!/usr/bin/env python3
import argparse
import random
import shutil
import time
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.backends import cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import tqdm


def main():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    arg = parser.add_argument
    arg('data', help='path to dataset')
    arg('--arch', default='resnet18',
        choices=model_names,
        help='model architecture: ' +
             ' | '.join(model_names) +
             ' (default: %(default)s)')
    arg('--workers', default=4, type=int,
        help='number of data loading workers (default: %(default)s)')
    arg('--epochs', default=90, type=int,
        help='number of total epochs to run')
    arg('--start-epoch', default=0, type=int,
        help='manual epoch number (useful on restarts)')
    arg('--batch-size', default=256, type=int,
        help='mini-batch size (default: %(default)s), this is the total '
             'batch size of all GPUs on the current node when '
             'using Data Parallel')
    arg('--lr', default=0.1, type=float,
        help='initial learning rate')
    arg('--momentum', default=0.9, type=float, help='momentum')
    arg('--weight-decay', default=1e-4, type=float,
        help='weight decay (default: %(default)s)')
    arg('--resume', default='', type=str,
        help='path to latest checkpoint (default: %(default)s)')
    arg('--evaluate', dest='evaluate', action='store_true',
        help='evaluate model on validation set')
    arg('--pretrained', action='store_true', help='use pre-trained model')
    arg('--seed', default=None, type=int, help='seed for initializing training')
    arg('--gpu', default=None, type=int, help='GPU id to use')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    print(f'Using model {args.arch} pretrained={args.pretrained}')
    model_cls = models.__dict__[args.arch]
    model = model_cls(pretrained=args.pretrained)

    if args.gpu is not None:
        print('Use GPU: {args.gpu} for training')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_acc1 = 0

    if args.resume:
        if not Path(args.resume).is_file():
            parser.error(f'checkpoint {args.resume} is not a file')
        print(f'loading checkpoint {args.resume}')
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')

    cudnn.benchmark = True

    data_root = Path(args.data)
    traindir = data_root / 'train'
    valdir = data_root / 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in tqdm.trange(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader: DataLoader, model: nn.Module, criterion, optimizer,
          epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix({
            'batch_t': f'{batch_time.avg:.3f}',
            'data_t': f'{data_time.avg:.3f}',
            'loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.3f}',
            'Acc@5': f'{top5.avg:.3f}',
        })


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm.tqdm(val_loader, desc='Validation')
        for i, (input, target) in enumerate(pbar):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'batch_t': f'{batch_time.avg:.3f}',
                'loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}',
                'Acc@5': f'{top5.avg:.3f}',
            })

        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg


def save_checkpoint(state, is_best, filename='net.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'net-best.pth')


class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, args):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
