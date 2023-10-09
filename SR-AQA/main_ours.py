import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from toe import TOE
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
import numpy as np
import network

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loading code
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.Resize([args.resize_size, args.resize_size]),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        T.Resize([args.resize_size, args.resize_size]),
        T.ToTensor(),
        normalize
    ])

    train_source_dataset = TOE(root=args.root, task=args.source, split='train', download=True,
                               transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = TOE(root=args.root, task=args.target, split='train', download=True,
                               transform=train_transform)
    
    # Get X% of target data
    if args.t_data_ratio == 1:
        t_data_select = random.randint(0, 9)
    elif args.t_data_ratio == 3:
        t_data_select = random.randint(0, 7)
    elif args.t_data_ratio == 5:
        t_data_select = random.randint(0, 5)
    elif args.t_data_ratio == 10:
        t_data_select = 0
    subset_range = [0, 495, 917, 1367, 1824,
                    2299, 2816, 3250, 3591, 4042, 4427]
    subset_data = torch.utils.data.Subset(
        train_target_dataset, range(subset_range[t_data_select], subset_range[t_data_select+args.t_data_ratio]))
    train_target_loader = DataLoader(subset_data, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = TOE(root=args.root, task=args.target,
                      split='test', download=True, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    print(len(train_source_loader))
    print(len(train_target_loader))
    print(len(val_loader))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    criterion = nn.MSELoss().to(device)
    num_classes = 1
    net = network.get_net(args, num_classes, criterion,
                          args.cont_proj_head, device=device)
    print(net)
    # define optimizer
    optimizer = SGD(net.parameters(), args.lr,
                    momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        net.load_state_dict(checkpoint)

    # start training
    best_mse = 100000.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, net, optimizer,
              epoch, args)

        # # evaluate on validation set
        if args.task_type == 'cp_reg':
            mae, mse = validate(val_loader, net, args, [
                                'CP'], device, None, None, None)
        elif args.task_type == 'gi_reg':
            mae, mse = validate(val_loader, net, args, [
                                'GI'], device, None, None, None)
                                
        # remember best mse and save checkpoint
        print("MSE {:.8f}, MAE: {:.6f}".format(mse, mae))
        if mse < best_mse:
            torch.save(net.state_dict(), logger.get_checkpoint_path(
            f'best'))
            best_mse = min(mse, best_mse)
            best_mae = mae
            print("updated best mse {:.8f}, corresponed mae: {:.6f}".format(
                best_mse, best_mae))

    print("best mse {:.8f}, corresponed mae: {:.6f}".format(
        best_mse, best_mae))


def train(train_source_iter, train_target_iter, model, optimizer,
          epoch, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    mse_losses = AverageMeter('MSE Loss', ':.6f')
    tl_losses_main = AverageMeter('TL Loss', ':.6f')
    scl_losses_main = AverageMeter('SCL Loss', ':.6f')
    total_losses = AverageMeter('Total Loss', ':.6f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, total_losses, mse_losses,
            tl_losses_main, scl_losses_main],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        if args.task_type == 'cp_reg':
            labels_s = labels_s[:, 1].to(device).float()/100.0
        else:
            labels_s = labels_s[:, 2].to(device).float()/4.0
        x_t, _ = next(train_target_iter)
        x_t = x_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(x=x_s, gts=labels_s, x_r=x_t, apply_fs=args.use_fs)

        outputs_index = 0
        main_loss = outputs[0]
        total_loss = main_loss
        outputs_index += 1

        if args.use_fs:
            if args.use_scl:
                scl_loss_main = outputs[outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.lambda_scl * scl_loss_main)
                scl_losses_main.update(scl_loss_main.item(), x_s.size(0))
            else:
                scl_loss_main = 0

            if args.use_tl:
                tl_loss_main = outputs[outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.lambda_tl * tl_loss_main)
                tl_losses_main.update(tl_loss_main.item(), x_s.size(0))
            else:
                tl_loss_main = 0

        mse_losses.update(main_loss.item(), x_s.size(0))
        total_losses.update(total_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        total_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, args, factors, device, accuracy, confusion_m, f1_score):
    batch_time = AverageMeter('Time', ':6.3f')

    # switch to evaluate mode
    model.eval()
    total_mse_loss = 0
    total_mae_loss = 0
    count = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            if args.task_type == 'cp_reg':
                target = target[:, 1].to(device).float()/100.0
            elif args.task_type == 'gi_reg':
                target = target[:, 2].to(device).float()/4.0
            elif args.task_type == 'class':
                target = target[:, 0].to(device).long()
            else:
                target = target.to(device)

            # compute output
            output = model(images)
            mae_loss = F.l1_loss(output, target)
            mse_loss = F.mse_loss(output, target)
            total_mae_loss += mae_loss
            total_mse_loss += mse_loss
            count += 1

            del images, target
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        mean_mae = total_mae_loss / float(count)
        mean_mse = total_mse_loss / float(count)

        return mean_mae, mean_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SR-AQA experiment')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='TEE')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--task_type', type=str, default="cp_reg",
                        choices=["cp_reg", "gi_reg"], help='choose the type of task')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default="cuda:0", help="device id to run")
    # training parameters
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 3)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters_per_epoch', default=100, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print_freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='log/debug',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'retrain'],
                        help="When phase is 'retrain', load the checkpoint.")
    parser.add_argument('--t_data_ratio', type=int, default=10,
                        help='use X percent of target data', choices=[1, 3, 5, 10])
    # model parameters
    parser.add_argument('--fs_layer', nargs='*', type=int, default=[1, 1, 1, 0, 0],
                        help='0: None, 1: UFS')
    parser.add_argument('--arch', type=str, default='network.ours.RDA',
                        help='Network architecture.')
    parser.add_argument('--cont_proj_head', type=int, default=256,
                        help='number of output channels of projector')
    parser.add_argument('--lambda_tl', type=float, default=0.0,
                        help='lambda for TL loss')
    parser.add_argument('--lambda_scl', type=float, default=0.0,
                        help='lambda for SCL loss')
    parser.add_argument('--use_fs', action='store_true', default=False,
                        help='Use UFS or not')
    parser.add_argument('--use_tl', action='store_true', default=False,
                        help='Use TL or not')
    parser.add_argument('--use_scl', action='store_true', default=False,
                        help='Use SCL or not')

    args = parser.parse_args()
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    
    for i in range(len(args.fs_layer)):
        if args.fs_layer[i] == 1:
            args.use_fs = True
    if args.lambda_tl > 0:
        args.use_tl = True
    if args.lambda_scl > 0:
        args.use_scl = True

    main(args)
