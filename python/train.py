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

import acsi


def build_masks(strings: List[str], theta_size: int) -> torch.BoolTensor:
    masks = []

    every = acsi.enumerate_masks(theta_size)
    sizes = every.sum(dim=1)

    for s in strings:
        if s.startswith('='):
            select = every.sum(dim=1) == int(s[1:])
            masks.append(every[select])
        else:
            mask = acsi.str2mask(s)[:theta_size]
            masks.append(mask.unsqueeze(0))

    return torch.cat(masks) if masks else None


def build_instance(settings: dict) -> Tuple[nn.Module, nn.Module]:
    # Simulator
    if settings['simulator'] == 'MLCP':
        simulator = acsi.MLCP()
    else:  # settings['simulator'] == 'SCLP'
        simulator = acsi.SLCP()

    theta, x = simulator.sample()

    theta_size = theta.numel()
    x_size = x.numel()

    # Masks
    masks = build_masks(settings['masks'], theta_size)

    # Model & Encoder
    if settings['encoder'] is None:
        encoder = nn.Flatten(-len(x.shape))
    else:
        a, b, c = settings['encoder']
        encoder = acsi.MLP(x.shape, output_size=c, num_layers=a, hidden_size=b)

    a, b = settings['model']
    if settings['arbitrary']:
        model = acsi.AMNRE(theta_size, x_size, masks=masks, encoder=encoder, num_layers=a, hidden_size=b)
    else:
        if masks is None:
            model = acsi.NRE(theta_size, x_size, encoder=encoder, num_layers=a, hidden_size=b)
        else:
            model = acsi.MNRE(masks, x_size, encoder=encoder, num_layers=a, hidden_size=b)

    # Load
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location='cpu')
        model.load_state_dict(weights)

    return simulator, model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-o', '--output', default=None, help='output files basename')
    parser.add_argument('-p', '--path', default='../products', help='output path')

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP'])
    parser.add_argument('-masks', nargs='+', default=[], help='marginalzation masks')
    parser.add_argument('-model', type=int, nargs=2, default=[10, 512], help='model architecture')
    parser.add_argument('-encoder', type=int, nargs=3, default=None, help='encoder architecture')
    parser.add_argument('-weights', default=None, help='warm-start weights')
    parser.add_argument('-arbitrary', default=False, action='store_true', help='arbitrary design')

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

    args = parser.parse_args()
    args.date = datetime.now().strftime('%y%m%d_%H%M%S')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulator & Model
    simulator, model = build_instance(vars(args))

    simulator.to(device)
    model.to(device)

    # Criterion(s)
    criterion = acsi.RELoss()

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
    trainset = acsi.LTEDataset(simulator, args.batch_size)

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
    if args.output is None:
        args.output = os.path.join(args.path, args.date).replace('\\', '/')
    else:
        args.path = os.path.dirname(args.output)

    if args.path:
        os.makedirs(args.path, exist_ok=True)

    ## Settings
    with open(args.output + '.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    ## Weights
    torch.save(model.cpu().state_dict(), args.output + '.pth')

    ## Stats
    df = pd.DataFrame(stats)
    df.to_csv(args.output + '.csv', index=False)
