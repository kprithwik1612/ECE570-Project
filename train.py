import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # For plotting loss and accuracy

from model import WideResnet
from cifar import get_train_loader, get_val_loader, OneHot
from label_guessor import LabelGuessor
from lr_scheduler import WarmupCosineLrScheduler
from ema import EMA

## args
parser = argparse.ArgumentParser(description='FixMatch Training')
parser.add_argument('--wresnet-k', default=2, type=int, help='width factor of wide resnet')
parser.add_argument('--wresnet-n', type=int, default=28, help='depth of wide resnet')
parser.add_argument('--n-classes', type=int, default=10, help='number of classes in dataset')
parser.add_argument('--n-labeled', type=int, default=40, help='number of labeled samples for training')
parser.add_argument('--n-epoches', type=int, default=1024, help='number of training epoches')
parser.add_argument('--batchsize', type=int, default=64, help='train batch size of labeled samples')
parser.add_argument('--mu', type=int, default=7, help='factor of train batch size of unlabeled samples')
parser.add_argument('--thr', type=float, default=0.95, help='pseudo label threshold')
parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024, help='number of training images for each epoch')
parser.add_argument('--lam-u', type=float, default=1., help='coefficient of unlabeled loss')
parser.add_argument('--ema-alpha', type=float, default=0.999, help='decay rate for ema module')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate for training')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--seed', type=int, default=-1, help='seed for random behaviors, no seed if negative')
args = parser.parse_args()

## global settings
if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

def set_model():
    model = WideResnet(args.n_classes, k=args.wresnet_k, n=args.wresnet_n)  # wresnet-28-2
    model.train()
    model.cuda()
    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss().cuda()
    return model, criteria_x, criteria_u

def train_one_epoch(
        model,
        criteria_x,
        criteria_u,
        optim,
        lr_schdlr,
        ema,
        dltrain_x,
        dltrain_u,
        lb_guessor,
        lambda_u,
        n_iters,
    ):
    total_loss = 0  # For tracking average loss per epoch
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)

    for it in range(n_iters):
        ims_x_weak, ims_x_strong, lbs_x = next(dl_x)
        ims_u_weak, ims_u_strong, lbs_u_real = next(dl_u)

        # Move tensors to GPU
        ims_x_weak, ims_x_strong, lbs_x = ims_x_weak.cuda(), ims_x_strong.cuda(), lbs_x.cuda()
        ims_u_weak, ims_u_strong, lbs_u_real = ims_u_weak.cuda(), ims_u_strong.cuda(), lbs_u_real.cuda()

        # Generate pseudo labels
        lbs_u, valid_u = lb_guessor(model, ims_u_weak)
        ims_u_strong, lbs_u, lbs_u_real = ims_u_strong[valid_u].cuda(), lbs_u.cuda(), lbs_u_real[valid_u].cuda()

        if ims_u_strong.size(0) > 0:
            ims_x_u = torch.cat([ims_x_weak, ims_u_strong], dim=0)
            lbs_x_u = torch.cat([lbs_x, lbs_u], dim=0)
            logits_x_u = model(ims_x_u)
            logits_x, logits_u = logits_x_u[:ims_x_weak.size(0)], logits_x_u[ims_x_weak.size(0):]
            loss_x, loss_u = criteria_x(logits_x, lbs_x), criteria_u(logits_u, lbs_u)
            loss = loss_x + lambda_u * loss_u
        else:
            logits_x = model(ims_x_weak)
            loss_x = criteria_x(logits_x, lbs_x)
            loss = loss_x

        total_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        # Print progress and current loss every 1% of the epoch
        if (it + 1) % (n_iters // 100) == 0:
            progress_percentage = (it + 1) / n_iters * 100
            print(f"Epoch Progress: {progress_percentage:.1f}% complete, Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / n_iters
    ema.update_buffer()
    return avg_loss



def evaluate(ema):
    ema.apply_shadow()
    ema.model.eval()
    dlval = get_val_loader(batch_size=128, num_workers=0, root='cifar10')

    correct = 0
    total = 0
    with torch.no_grad():
        for ims, lbs in dlval:
            ims, lbs = ims.cuda(), lbs.cuda()
            outputs = ema.model(ims)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbs).sum().item()
            total += lbs.size(0)

    acc = correct / total
    ema.restore()
    return acc

def train():
    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize
    model, criteria_x, criteria_u = set_model()
    dltrain_x, dltrain_u = get_train_loader(args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled)
    lb_guessor = LabelGuessor(thresh=args.thr)
    ema = EMA(model, args.ema_alpha)

    wd_params, non_wd_params = [], []
    for param in model.parameters():
        (wd_params if len(param.size()) != 1 else non_wd_params).append(param)

    optim = torch.optim.SGD([{'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}],
                            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(optim, max_iter=n_iters_per_epoch * args.n_epoches, warmup_iter=0)

    train_losses, val_accuracies = [], []
    print('start to train')

    for e in range(args.n_epoches):
        model.train()
        print(f'Starting epoch {e+1}')
        avg_loss = train_one_epoch(model, criteria_x, criteria_u, optim, lr_schdlr, ema, dltrain_x, dltrain_u, lb_guessor, args.lam_u, n_iters_per_epoch)
        acc = evaluate(ema)

        # Log and print epoch results
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        print(f'Epoch [{e+1}/{args.n_epoches}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

        # Plot training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    train()
