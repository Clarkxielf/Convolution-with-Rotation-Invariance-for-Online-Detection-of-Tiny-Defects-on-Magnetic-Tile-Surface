import argparse
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from shutil import rmtree
from RFMNet.RFMNet_L3C2 import *
from Data_loader import data_loader
from Confusion_matrix_data2 import ConfusionMatrix

model_names = ['rfmnet']

parser = argparse.ArgumentParser(description='PyTorch Dataset Training')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpointRFMNet_L3C2', type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=True,
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='./checkpointRFMNet_L3C2/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: ./checkpointRFMNet_L3C2/model_best.pth.tar)')

parser.add_argument('-data', default='../Data_processing_data2/Result_Split_data',
                    type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='rfmnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: rfmnet)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true', default=True,
                    help='use pin memory')
parser.add_argument('--print-freq', '-f', default=20, type=int, metavar='N',
                    help='print frequency (default: 20)')


best_prec1 = 0.0

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, args.save_dir+('/model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    SEED = 2022
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if args.evaluate is False:
        if os.path.exists(args.save_dir):
            # 如果文件夹存在，则先删除原文件夹再重新创建
            rmtree(args.save_dir)
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, 'log.txt'), 'a')

    print_log('save path : {}'.format(args.save_dir), log)
    print_log(args, log)

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)


    if args.arch == 'rfmnet':
        model = rfmnet()
    else:
        raise NotImplementedError

    print_log("=> network :\n {}".format(model), log)

    # model = torch.nn.DataParallel(model)
    # use cuda
    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader, test_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        CM = ConfusionMatrix(num_classes=3, labels=['MT_Blowhole', 'MT_Crack', 'MT_Free'])
        validate(test_loader, model, criterion, args.print_freq, log, CM)
        return

    else:
        train_top1, train_losses, val_top1, val_losses = [], [], [], []
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr)

            # train for one epoch
            top1, train_loss = train(train_loader, model, criterion, optimizer, epoch, args.print_freq, log)
            train_top1.append(top1)
            train_losses.append(train_loss)

            # evaluate on validation set
            prec1, val_loss = validate(test_loader, model, criterion, args.print_freq, log, None)
            val_top1.append(prec1)
            val_losses.append(val_loss)

            # remember the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best)

        sio.savemat(args.save_dir+'/Acc_Loss.mat', {'train_top1': np.stack(train_top1),
                                                    'train_losses': np.stack(train_losses),
                                                    'val_top1': np.stack(val_top1),
                                                    'val_losses': np.stack(val_losses),})

        Color = ['#E05C22', '#00A695', '#CF4AE1', ]
        plt.figure(figsize=(10, 4))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_top1, '-', color=Color[0], label='train')
        plt.plot(range(1, args.epochs + 1), val_top1, '--', color=Color[1], label='val')
        plt.ylabel('Accuracy(%)')
        plt.xlabel('Epochs')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.epochs + 1), train_losses, '-', color=Color[0], label='train')
        plt.plot(range(1, args.epochs + 1), val_losses, '--', color=Color[1], label='val')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')

        plt.savefig(args.save_dir + '/Acc_Loss.jpg', dpi=200)
        plt.show()




def train(train_loader, model, criterion, optimizer, epoch, print_freq, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        [prec1, ] = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1), log)

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, print_freq, log, CM):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, ] = accuracy(output.data, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if args.evaluate:
                CM.update(preds=output.data.argmax(dim=-1), labels=target)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1), log)

    print_log(' * Prec@1 {top1.avg:.3f}'.format(top1=top1), log)

    if args.evaluate:
        CM.summary()
        CM.plot(args.save_dir + '/ConfusionMatrix.jpg')

    return top1.avg, losses.avg


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()