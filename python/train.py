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


def build_masks(strings: List[str], theta_size: int) -> torch.BoolTensor:
    if not strings:
        return None

    masks = []

    every = amsi.enumerate_masks(theta_size)
    sizes = every.sum(dim=1)

    for s in strings:
        if s.startswith('='):
            select = every.sum(dim=1) == int(s[1:])
            masks.append(every[select])
        else:
            mask = amsi.str2mask(s)[:theta_size]
            masks.append(mask.unsqueeze(0))

    return torch.cat(masks)


def build_instance(settings: dict) -> Tuple[amsi.Simulator, nn.Module]:
    # Simulator
    if settings['simulator'] == 'GW':
        with h5py.File(settings['samples'], 'r') as f:
            simulator = amsi.GW(basis=f['basis'][:])
    elif settings['simulator'] == 'MLCP':
        simulator = amsi.MLCP()
    else:  # settings['simulator'] == 'SCLP'
        simulator = amsi.SLCP()

    # Placeholders
    theta, x = simulator.sample()

    theta_size = theta.numel()
    x_size = x.numel()

    # Moments
    if settings['weights'] is None:
        moments = torch.zeros(theta_size), torch.ones(theta_size)
    else:
        theta = simulator.prior.sample((2 ** 16,))
        moments = torch.mean(theta, dim=0), torch.std(theta, dim=0)

    # Model & Encoder
    if settings['encoder'] is None:
        encoder = nn.Flatten(-len(x.shape))
    else:
        encoder = amsi.MLP(x.shape, **settings['encoder'])
        x_size = encoder.output_size

    default_model = {
        'num_layers': 10,
        'hidden_size': 256,
        'activation': 'SELU',
    }
    default_model.update(settings['model'])
    settings['model'] = default_model

    if settings['arbitrary']:
        model = amsi.AMNRE(theta_size, x_size, encoder=encoder, moments=moments, **settings['model'])
    else:
        masks = build_masks(settings['masks'], theta_size)

        if masks is None:
            model = amsi.NRE(theta_size, x_size, encoder=encoder, moments=moments, **settings['model'])
        else:
            model = amsi.MNRE(masks, x_size, encoder=encoder, moments=moments, **settings['model'])

    ## Weights
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location='cpu')
        model.load_state_dict(weights)

    # Device
    simulator.to(settings['device'])
    model.to(settings['device'])

    return simulator, model


def from_settings(filename: str) -> Tuple[amsi.Simulator, nn.Module]:
    with open(filename) as f:
        settings = json.load(f)
        settings['weights'] = filename.replace('.json', '.pth')

    return build_instance(settings)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW'])
    parser.add_argument('-samples', default=None, help='samples file (H5)')
    parser.add_argument('-model', type=json.loads, default={}, help='model architecture')
    parser.add_argument('-encoder', type=json.loads, default=None, help='encoder architecture')
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

    parser.add_argument('-target', default=None, help='target network settings file (JSON)')
    parser.add_argument('-coef', type=float, nargs=2, default=(1e-3, 1e-3), help='distillation losses coefficients')

    parser.add_argument('-o', '--output', default='../products/models/out.pth', help='output file (PTH)')

    args = parser.parse_args()
    args.date = datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')

    # Simulator & Model
    simulator, model = build_instance(vars(args))

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
            masks = build_masks(args.masks, theta_size)
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

    # Dataset
    if args.samples:
        trainset = amsi.OfflineLTEDataset(args.samples, batch_size=args.batch_size, device=args.device)
    else:
        trainset = amsi.OnlineLTEDataset(simulator, args.batch_size)

    # Target network
    if args.target is None or type(model) is amsi.NRE:
        targetnet = None
    else:
        _, targetnet = from_settings(args.target)
        targetnet.to(args.device)

    # Training
    stats = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        losses = []

        start = time()

        for theta, theta_prime, x in islice(trainset, args.per_epoch):
            theta.requires_grad = True

            if args.arbitrary:
                if model.hyper is None:
                    mask = mask_sampler(theta.shape[:1])
                else:
                    mask = mask_sampler()
                    model[mask]
                    mask = None

            z = model.encoder(x)

            if args.arbitrary:
                ratio = model(theta, z, mask)
                ratio_prime = model(theta_prime, z, mask)
            else:
                ratio = model(theta, z)
                ratio_prime = model(theta_prime, z)

            l = criterion(ratio, ratio_prime)

            if targetnet is not None:
                if args.arbitrary:
                    theta_mix = torch.where(mask, theta, theta_prime)
                else:
                    theta_mix = torch.stack([
                        torch.where(mask, theta, theta_prime)
                        for mask, _ in model
                    ], dim=-2)

                target_ratio = targetnet(theta_mix, targetnet.encoder(x))

                rdl = args.coef[0] * rd_loss(ratio, target_ratio)
                sdl = args.coef[1] * sd_loss(theta, ratio, target_ratio)

                l = torch.stack([l, rdl, sdl])

            losses.append(l.tolist())

            optimizer.zero_grad()
            l.sum().backward()
            optimizer.step()

        end = time()

        ## Stats
        losses = torch.tensor(losses)

        stats.append({
            'epoch': epoch,
            'time': end - start,
            'lr': optimizer.param_groups[0]['lr'],
            'mean': losses.mean(dim=0).tolist(),
            'std': losses.std(dim=0).tolist(),
        })

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
        json.dump(vars(args), f, indent=4)

    ## Stats
    df = pd.DataFrame(stats)
    df.to_csv(args.output.replace('.pth', '.csv'), index=False)
