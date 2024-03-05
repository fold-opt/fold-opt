#!/usr/bin/env python3

# import waitGPU
# waitGPU.wait(interval=10, nproc=0, ngpu=1)
# import setGPU

import argparse

from cli import add_all_parsers, set_defaults
import time
import torch
import logger

from utils import create_experiment, get_optimizer, set_seed,\
    update_optimizer, load_optimizer
from data.main import get_loaders
from epoch import train, test

from losses.main import get_loss
from models.main import get_model, load_model

from setproctitle import setproctitle

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def run(args):
    print('args')
    tag = 'bamos.smooth-topk.seed={}.{}'.format(args.seed, args.dataset)
    if args.dataset == 'cifar100':
        tag += '.noise={}'.format(args.noise_labels)
    elif args.dataset == 'imagenet':
        tag += '-{}'.format(args.train_size)
    setproctitle(tag)

    set_seed(args.seed)
    xp = create_experiment(args)
    train_loader, val_loader, test_loader = get_loaders(args)
    loss = get_loss(xp, args)

    model = get_model(args)
    if args.load_model:
        load_model(model, args.load_model)

    if args.cuda:
        if args.parallel_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            torch.cuda.set_device(args.device)
            model.cuda()
        loss.cuda()

    optimizer = get_optimizer(model, args.mu, args.lr_0, xp)
    if args.load_optimizer:
        load_optimizer(optimizer, args.load_optimizer, args.lr_0)

    with logger.stdout_to("{}_log.txt".format(args.out_name)):
        clock = -time.time()
        for idx in range(args.epochs):

            xp.Epoch.update(1).log()
            optimizer = update_optimizer(args.lr_schedule, optimizer,
                                         model, loss, xp)

            xp.Learning_Rate.update().log()
            xp.Mu.update().log()
            xp.Temperature.update().log()

            train(model, loss, optimizer, train_loader, xp, args, idx)
            test(model, loss, val_loader, xp, args)

        test(model, loss, test_loader, xp, args)
        clock += time.time()

        print("\nEvaluation time:  \t {0:.2g} min".format(clock * 1. / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    set_defaults(args)
    run(args)
