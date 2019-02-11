#!/usr/bin/env python3
import argparse
import random
import shutil
import time
from pathlib import Path
import warnings

import mlflow
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.backends import cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import tqdm

from imagenet import lr_schedule


def main():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    arg = parser.add_argument
    arg('data', help='path to dataset')
    arg('--name', help='mlflow run name')
    arg('--exp-name', default='default', help='mlflow experiment name')
    arg('--arch', default='resnet18',
        choices=model_names,
        help='model architecture: ' +
             ' | '.join(model_names) +
             ' (default: %(default)s)')
    arg('--input-size', default=224, type=int, help='input size for the network')
    arg('--predict-size', default=256, type=int,
        help='at test time, resize to that size, then crop to --input-size')
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
    arg('--lr-schedule', default='imagenet_schedule',
        help='schedule for the learning rate (see schedules.py)')
    arg('--momentum', default=0.9, type=float, help='momentum')
    arg('--weight-decay', default=1e-4, type=float,
        help='weight decay (default: %(default)s)')
    arg('--resume', help='path to latest checkpoint')
    arg('--evaluate', action='store_true',
        help='evaluate model on validation set')
    arg('--pretrained', action='store_true', help='use pre-trained model')
    arg('--seed', type=int, help='seed for initializing training')
    arg('--device', type=str, default='cuda',
        help='device to use, "cuda" to use all GPUs, '
             '"cuda:0" to use a specific GPU,'
             '"cpu" to run on CPU.')
    arg('--train-limit', type=int, help='limit train dataset size')
    arg('--valid-limit', type=int, help='limit valid dataset size')
    args = parser.parse_args()

    params = vars(args)
    run_name = params.pop('name')
    mlflow.set_experiment(experiment_name=args.exp_name)
    mlflow.start_run(run_name=run_name)
    for k, v in params.items():
        mlflow.log_param(k, v)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print(f'Using model {args.arch} pretrained={args.pretrained}')
    model = create_model(args.arch)

    if args.device == 'cuda':
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
    model.to(device=args.device)

    criterion = nn.CrossEntropyLoss().to(device=args.device)
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
    train_dir = data_root / 'train'
    valid_dir = data_root / 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        loader=pil_loader,
    )
    sample_imagefolder(train_dataset, args.train_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    valid_dataset = datasets.ImageFolder(
        valid_dir, transforms.Compose([
            transforms.Resize(args.predict_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize,
        ]),
        loader=pil_loader,
    )
    sample_imagefolder(valid_dataset, args.valid_limit)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(valid_loader, model, criterion, args)
        return

    adjust_learning_rate = getattr(lr_schedule, args.lr_schedule)

    pbar = tqdm.trange(args.start_epoch, args.epochs)
    for epoch in pbar:
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(valid_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        pbar.set_postfix({'acc1': f'{acc1:.3f}'})

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
    for input, target in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device=args.device, non_blocking=True)
        target = target.to(device=args.device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

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

    mlflow.log_metric('train_loss', losses.avg)
    mlflow.log_metric('train_top1', top1.avg)
    mlflow.log_metric('train_top5', top5.avg)


def validate(valid_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm.tqdm(valid_loader, desc='Validation')
        for input, target in pbar:
            input = input.to(device=args.device, non_blocking=True)
            target = target.to(device=args.device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'batch_t': f'{batch_time.avg:.3f}',
                'loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}',
                'Acc@5': f'{top5.avg:.3f}',
            })

    mlflow.log_metric('valid_loss', losses.avg)
    mlflow.log_metric('valid_top1', top1.avg)
    mlflow.log_metric('valid_top5', top5.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='net.pth'):
    # TODO make it an mlflow artifact
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'net-best.pth')


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_model(arch: str, pretrained=False, num_classes=1000) -> nn.Module:
    model_cls = models.__dict__[arch]
    model = model_cls(pretrained=pretrained, num_classes=num_classes)
    assert hasattr(model, 'avgpool')
    model.avgpool = AvgPool()
    return model


class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


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


def sample_imagefolder(dataset: datasets.ImageFolder, limit: int = None):
    if limit and len(dataset) > limit:
        rng = np.random.RandomState(42)
        dataset.samples = [
            dataset.samples[idx]
            for idx in rng.choice(len(dataset.samples), limit, replace=False)]


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with warnings.catch_warnings(), open(path, 'rb') as f:
        warnings.filterwarnings('ignore', category=UserWarning)
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    main()
