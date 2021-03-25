#!/usr/bin/env python

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

    return torch.cat(masks) if masks else None


def build_instance(settings: dict) -> Tuple[nn.Module, nn.Module]:
    # Simulator
    if settings['simulator'] == 'MLCP':
        simulator = amsi.MLCP()
    else:  # settings['simulator'] == 'SCLP'
        simulator = amsi.SLCP()

    # Placeholders
    theta, x = simulator.sample()

    theta_size = theta.numel()
    x_size = x.numel()

    # Masks
    masks = build_masks(settings['masks'], theta_size)

    # Model & Encoder
    if settings['encoder'] is None:
        encoder = nn.Flatten(-len(x.shape))
    else:
        encoder = amsi.MLP(x.shape, **settings['encoder'])
        x_size = encoder.output_size

    if settings['model'] is None:
        settings['model'] = {
            'num_layers': 10,
            'hidden_size': 256,
            'activation': 'SELU',
        }

    if settings['arbitrary']:
        model = amsi.AMNRE(theta_size, x_size, masks=masks, encoder=encoder, **settings['model'])
    else:
        if masks is None:
            model = amsi.NRE(theta_size, x_size, encoder=encoder, **settings['model'])
        else:
            model = amsi.MNRE(masks, x_size, encoder=encoder, **settings['model'])

    # Device
    simulator.to(settings['device'])
    model.to(settings['device'])

    # Load
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location=settings['device'])
        model.load_state_dict(weights)

    return simulator, model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP'])
    parser.add_argument('-masks', nargs='+', default=[], help='marginalzation masks')
    parser.add_argument('-model', type=json.loads, default=None, help='model architecture')
    parser.add_argument('-encoder', type=json.loads, default=None, help='encoder architecture')
    parser.add_argument('-arbitrary', default=False, action='store_true', help='arbitrary design')
    parser.add_argument('-weights', default=None, help='warm-start weights')

    parser.add_argument('-samples', default=None, help='samples file (H5)')

    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('-per-epoch', type=int, default=256, help='batches per epoch')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weight-decay', type=float, default=0., help='weight decay')
    parser.add_argument('-amsgrad', type=bool, default=True, help='AMS gradient')
    parser.add_argument('-patience', type=int, default=10, help='scheduler patience')
    parser.add_argument('-threshold', type=float, default=1e-2, help='scheduler threshold')
    parser.add_argument('-factor', type=float, default=1e-1, help='scheduler factor')
    parser.add_argument('-min-lr', type=float, default=1e-6, help='minimum learning rate')

    parser.add_argument('-o', '--output', default='../products/models/out.pth', help='output file (PTH)')

    args = parser.parse_args()
    args.date = datetime.now().strftime('%y%m%d_%H%M%S')

    # Simulator & Model
    simulator, model = build_instance(vars(args))

    # Criterion(s)
    criterion = amsi.RELoss()

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
        trainset = amsi.OfflineLTEDataset(args.samples, simulator.prior, args.batch_size)
    else:
        trainset = amsi.OnlineLTEDataset(simulator, args.batch_size)

    # Training
    stats = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        losses = []

        start = time()

        for theta, theta_prime, x in islice(trainset, args.per_epoch):
            ratio = model(theta, x)
            ratio_prime = model(theta_prime, x)

            l = criterion(ratio, ratio_prime)

            losses.append(l.item())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        end = time()

        ## Stats
        losses = torch.tensor(losses)

        stats.append({
            'epoch': epoch,
            'time': end - start,
            'lr': optimizer.param_groups[0]['lr'],
            'mean': losses.mean().item(),
            'std': losses.std().item(),
        })

        ## Scheduler
        scheduler.step(stats[-1]['mean'])

    # Output
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ## Weights
    torch.save(model.cpu().state_dict(), args.output)

    ## Settings
    with open(args.output.replace('.pth', '.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    ## Stats
    df = pd.DataFrame(stats)
    df.to_csv(args.output.replace('.pth', '.csv'), index=False)
