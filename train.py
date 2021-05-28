#!/usr/bin/env python

import h5py
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from itertools import count, islice
from time import time
from tqdm import tqdm
from typing import List, Tuple

import amsi


def build_encoder(input_size: torch.Size, name: str = None, **kwargs) -> tuple:
    flatten = nn.Flatten(-len(input_size))

    if name == 'MLP':
        net = amsi.MLP(input_size.numel(), **kwargs)
        return nn.Sequential(flatten, net), net.output_size
    elif name == 'ResNet':
        net = amsi.ResNet(input_size.numel(), **kwargs)
        return nn.Sequential(flatten, net), net.output_size
    else:
        return flatten, input_size.numel()


def build_instance(settings: dict) -> tuple:
    # Simulator
    if settings['simulator'] == 'GW':
        simulator = amsi.GW()
    elif settings['simulator'] == 'HH':
        simulator = amsi.HH()
    elif settings['simulator'] == 'MLCP':
        simulator = amsi.MLCP()
    else:  # settings['simulator'] == 'SCLP'
        simulator = amsi.SLCP()

    simulator.to(settings['device'])

    # Dataset
    if settings['samples'] is None:
        dataset = amsi.OnlineLTEDataset(simulator, settings['batch_size'])
    else:
        dataset = amsi.OfflineLTEDataset(settings['samples'], batch_size=settings['batch_size'], device=settings['device'])

    # Placeholders
    if settings['samples'] is None:
        theta, x = simulator.sample()
    else:
        theta, x = dataset[0]

        if theta is None:
            theta = simulator.prior.sample()

    theta_size = theta.numel()
    x_size = x.numel()

    # Moments
    if settings['weights'] is None:
        theta = simulator.prior.sample((2 ** 18,))
        moments = torch.mean(theta, dim=0), torch.std(theta, dim=0)
    else:
        moments = torch.zeros(theta_size), torch.ones(theta_size)

    # Model & Encoder
    encoder, x_size = build_encoder(x.shape, **settings['encoder'])

    model_args = settings['model'].copy()
    model_args['encoder'] = encoder
    model_args['moments'] = moments

    if settings['arbitrary']:
        model_args['hyper'] = settings['hyper']
        model = amsi.AMNRE(theta_size, x_size, **model_args)
    else:
        masks = amsi.list2masks(settings['masks'], theta_size)

        if len(masks) == 0:
            model = amsi.NRE(theta_size, x_size, **model_args)
        else:
            model = amsi.MNRE(masks, x_size, **model_args)

    ## Weights
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location='cpu')
        model.load_state_dict(weights)

    model.to(settings['device'])

    return simulator, dataset, model


def load_settings(filename: str) -> dict:
    with open(filename) as f:
        settings = json.load(f)
        settings['weights'] = filename.replace('.json', '.pth')

    return settings


def routine(dataset) -> Tuple[float, torch.Tensor]:
    losses = []

    start = time()

    for theta, theta_prime, x in islice(dataset, args.per_epoch):
        theta.requires_grad = True
        z = model.encoder(x)

        if args.arbitrary:
            if model.hyper is None:
                mask = mask_sampler(theta.shape[:1])
            else:
                mask = mask_sampler()
                model[mask]
                mask = None

            ratio = model(theta, z, mask)
            ratio_prime = model(theta_prime, z, mask)
        else:
            ratio = model(theta, z)
            ratio_prime = model(theta_prime, z)

        l = criterion(ratio, ratio_prime)

        if targetnet is not None:
            target_z = targetnet.encoder(x)
            target_ratio = targetnet(theta, target_z)
            target_ratio_prime = targetnet(theta_prime, target_z)

            rdl = args.coef[0] * rd_loss(ratio_prime, target_ratio_prime)
            irdl = args.coef[0] * rd_loss(-ratio, -target_ratio)
            sdl = args.coef[1] * sd_loss(theta, ratio, target_ratio)

            l = torch.stack([l, rdl, irdl, sdl])

        losses.append(l.tolist())

        if l.requires_grad:
            optimizer.zero_grad()
            l.sum().backward()
            optimizer.step()

    end = time()

    return end - start, torch.tensor(losses)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW', 'HH'])
    parser.add_argument('-samples', default=None, help='samples file (H5)')
    parser.add_argument('-model', type=json.loads, default={}, help='model architecture')
    parser.add_argument('-hyper', type=json.loads, default=None, help='hypernet architecture')
    parser.add_argument('-encoder', type=json.loads, default={}, help='encoder architecture')
    parser.add_argument('-masks', nargs='+', default=[], help='marginalzation masks')
    parser.add_argument('-arbitrary', default=False, action='store_true', help='arbitrary design')
    parser.add_argument('-weights', default=None, help='warm-start weights')

    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('-per-epoch', type=int, default=256, help='batches per epoch')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weight-decay', type=float, default=0., help='weight decay')
    parser.add_argument('-amsgrad', type=bool, default=True, help='AMS gradient')
    parser.add_argument('-patience', type=int, default=10, help='scheduler patience')
    parser.add_argument('-threshold', type=float, default=1e-2, help='scheduler threshold')
    parser.add_argument('-factor', type=float, default=2e-1, help='scheduler factor')
    parser.add_argument('-min-lr', type=float, default=1e-6, help='minimum learning rate')

    parser.add_argument('-valid', default=None, help='validation samples file (H5)')

    parser.add_argument('-target', default=None, help='target network settings file (JSON)')
    parser.add_argument('-coef', type=float, nargs=2, default=(1e-3, 1e-3), help='distillation losses coefficients')

    parser.add_argument('-o', '--output', default='../products/models/out.pth', help='output file (PTH)')

    args = parser.parse_args()
    args.date = datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')

    settings = vars(args)

    # Simulator & Model
    simulator, trainset, model = build_instance(settings)

    ## Arbitrary masks
    if args.arbitrary:
        theta_size = simulator.prior.sample().numel()

        if not args.masks:
            args.masks.append('uniform')

        if args.masks[0] == 'poisson':
            mask_sampler = amsi.PoissonMask(theta_size)
        elif args.masks[0] == 'uniform':
            mask_sampler = amsi.UniformMask(theta_size)
        else:
            masks = amsi.list2masks(args.masks, theta_size)
            mask_sampler = amsi.SelectionMask(masks)

        mask_sampler.to(args.device)

    # Criterion(s)
    criterion = amsi.RELoss()
    rd_loss = amsi.RDLoss()
    sd_loss = amsi.SDLoss()

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.factor,
        patience=args.patience,
        threshold=args.threshold,
        min_lr=args.min_lr,
    )

    # Validation
    if args.valid is not None:
        validset = amsi.OfflineLTEDataset(args.valid, batch_size=args.batch_size, device=args.device)

    # Target network
    if args.target is None or type(model) is amsi.NRE:
        targetnet = None
    else:
        _, _, targetnet = build_instance(load_settings(args.target))
        targetnet.to(args.device)

    # Training
    stats = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        timing, losses = routine(trainset)

        stats.append({
            'epoch': epoch,
            'time': timing,
            'lr': optimizer.param_groups[0]['lr'],
            'mean': losses.mean(dim=0).tolist(),
            'std': losses.std(dim=0).tolist(),
        })

        if args.valid is not None:
            with torch.no_grad():
                _, v_losses = routine(validset)

            stats[-1]['v_mean'] = v_losses.mean(dim=0).tolist()
            stats[-1]['v_std'] = v_losses.std(dim=0).tolist()

        ## Scheduler
        scheduler.step(losses.mean())

    # Output
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ## Weights
    if hasattr(model, 'clear'):
        model.clear()

    torch.save(model.cpu().state_dict(), args.output)

    ## Settings
    with open(args.output.replace('.pth', '.json'), 'w') as f:
        json.dump(settings, f, indent=4)

    ## Stats
    df = pd.DataFrame(stats)
    df.to_csv(args.output.replace('.pth', '.csv'), index=False)
