#!/usr/bin/env python

import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from itertools import count, islice
from time import time
from tqdm import tqdm
from typing import Tuple

from acsi import *


def instance(args) -> Tuple[nn.Module, nn.Module]:
    # Simulator
    if args['simulator'] == 'MLCP':
        simulator = MLCP()
    else:  # args['simulator'] == 'SCLP'
        simulator = SLCP()

    theta, x = simulator.sample()

    # Masks
    D = theta.size(0)

    if args['masks'] == 'tri-right':
        masks = [[False] * (D - i) + [True] * i for i in range(1, D + 1)]
    elif args['masks'] == 'tri-left':
        masks = [[True] * i + [False] * (D - i) for i in range(1, D + 1)]
    else:  # args['masks'] == 'pairs'
        masks = []

        for i in range(D):
            for j in range(i + 1):
                masks.append([k in [i, j] for k in range(D)])

        masks.append([True] * D)

    masks = torch.tensor(masks)

    # Model & Encoder
    x_size = x.numel()

    if args['encoder'] is None:
        encoder = None
    else:
        a, b, c = args['encoder']
        encoder = MLP(x_size, output_size=c, num_layers=a, hidden_size=b)

    a, b = args['model']
    model = MNRE(masks, x_size=x_size, encoder=encoder, num_layers=a, hidden_size=b)

    return simulator, model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-o', '--output', default=None, help='output files basename')
    parser.add_argument('-p', '--path', default='../products', help='output path')

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP'])
    parser.add_argument('-masks', default='tri-right', choices=['tri-left', 'tri-right', 'pairs'])
    parser.add_argument('-model', type=int, nargs=2, default=[10, 512], help='model architecture')
    parser.add_argument('-encoder', type=int, nargs=3, default=None, help='encoder architecture')

    parser.add_argument('-weights', default=None, help='warm-start weights')

    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('-per-epoch', type=int, default=256, help='batches per epoch')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weight-decay', type=float, default=0., help='weight decay')
    parser.add_argument('-amsgrad', type=bool, default=True, help='AMS gradient')
    parser.add_argument('-patience', type=int, default=10, help='scheduler patience')
    parser.add_argument('-threshold', type=float, default=1e-2, help='scheduler threshold')
    parser.add_argument('-factor', type=float, default=1e-1, help='scheduler factor')
    parser.add_argument('-min-lr', type=float, default=1e-6, help='scheduler minimum learning rate')

    parser.add_argument('-alpha', type=float, nargs=3, default=[1., 0., 0.], help='loss weights')

    args = parser.parse_args()
    args.date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Simulator & Model
    simulator, model = instance(vars(args))

    simulator.to(device)
    model.to(device)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # Dataset
    trainset = LTEDataset(simulator, args.batch_size)

    # Criterion(s)
    criterion = RELoss()
    rd_loss = RDLoss()
    sd_loss = SDLoss()

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

    # Training
    _, full = model[-1]
    alpha, beta, gamma = args.alpha

    stats = []

    for epoch in tqdm(range(args.epochs)):
        losses = []

        start = time()

        for theta, theta_prime, x in islice(trainset, args.per_epoch):
            loss = []

            if alpha > 0.:
                ratio = model(theta, x)
                ratio_prime = model(theta_prime, x)

                l = criterion(ratio, ratio_prime)
                loss.append(alpha * l)

            if beta > 0. or gamma > 0.:
                with torch.no_grad():
                    z = model.encoder(x)

                rdl, sdl = 0., 0. # ratio & score distillation loss

                for mask, nre in model:
                    if torch.all(mask):
                        continue

                    mix = torch.where(mask, theta, theta_prime)
                    mix.requires_grad = True

                    ratio = nre(mix[..., mask], z)
                    target_ratio = full(mix, z)

                    if beta > 0.:
                        rdl = rdl + rd_loss(ratio, target_ratio.detach())

                    if gamma > 0.:
                        score = torch.autograd.grad(
                            ratio, mix,
                            torch.ones_like(ratio),
                            create_graph=True
                        )[0]

                        target_score = torch.autograd.grad(
                            target_ratio, mix,
                            torch.ones_like(target_ratio)
                        )[0]

                        sdl = sdl + sd_loss(score, target_score.detach())

                if beta > 0.:
                    loss.append(beta * rdl)

                if gamma > 0.:
                    loss.append(gamma * sdl)

            loss = torch.stack(loss)
            losses.append(loss.tolist())

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        end = time()

        ## Stats
        losses = torch.tensor(losses)

        stats.append({
            'epoch': epoch + 1,
            'time': end - start,
            'lr': optimizer.param_groups[0]['lr'],
            'mean': losses.mean(dim=0).tolist(),
            'std': losses.std(dim=0).tolist(),
        })

        ## Scheduler
        scheduler.step(losses.mean())

    # Output
    if args.output is None:
        args.output = os.path.join(args.path, args.date)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ## Setup
    with open(args.output + '.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    ## Weights
    torch.save(model.cpu().state_dict(), args.output + '.pth')

    ## Stats
    df = pd.DataFrame(stats)
    df.to_csv(args.output + '.csv', index=False)
