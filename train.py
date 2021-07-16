#!/usr/bin/env python

import h5py
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from contextlib import nullcontext
from datetime import datetime
from itertools import islice
from time import time
from tqdm import tqdm
from typing import List, Tuple

import amnre


def build_encoder(input_size: torch.Size, name: str = None, **kwargs) -> tuple:
    flatten = nn.Flatten(-len(input_size))

    if name == 'MLP':
        net = amnre.MLP(input_size.numel(), **kwargs)
        return nn.Sequential(flatten, net), net.output_size
    elif name == 'ResNet':
        net = amnre.ResNet(input_size.numel(), **kwargs)
        return nn.Sequential(flatten, net), net.output_size
    else:
        return flatten, input_size.numel()


def build_instance(settings: dict) -> tuple:
    # Simulator
    if settings['simulator'] == 'GW':
        simulator = amnre.GW()
    elif settings['simulator'] == 'HH':
        simulator = amnre.HH()
    elif settings['simulator'] == 'MLCP':
        simulator = amnre.MLCP()
    else:  # settings['simulator'] == 'SCLP'
        simulator = amnre.SLCP()

    simulator.to(settings['device'])

    # Dataset
    if settings['samples'] is None:
        dataset = amnre.OnlineDataset(simulator, batch_size=settings['bs'])
        theta, x = simulator.sample()
    else:
        dataset = amnre.OfflineDataset(settings['samples'], batch_size=settings['bs'], device=settings['device'])
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
        model = amnre.AMNRE(theta_size, x_size, **model_args)
    else:
        masks = amnre.list2masks(settings['masks'], theta_size, settings['filter'])

        if len(masks) == 0:
            model = amnre.NRE(theta_size, x_size, **model_args)
        else:
            model = amnre.MNRE(masks, x_size, **model_args)

    ## Weights
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location='cpu')
        model.load_state_dict(weights)

    model.to(settings['device'])

    return simulator, dataset, model


def load_settings(filename: str) -> dict:
    with open(filename) as f:
        settings = json.load(f)

    return settings


def load_model(filename: str) -> nn.Module:
    settings = load_settings(filename.replace('.pth', '.json'))
    settings['weights'] = filename

    _, _, model = build_instance(settings)

    return model


class Dummy(nn.Module):
    def __getitem__(self, idx):
        return None

    def forward(self, *args):
        return None

    def encoder(self, *args):
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train neural ratio estimator')

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW', 'HH'])
    parser.add_argument('-samples', default=None, help='samples file (H5)')
    parser.add_argument('-model', type=json.loads, default={}, help='model architecture')
    parser.add_argument('-hyper', type=json.loads, default=None, help='hypernet architecture')
    parser.add_argument('-encoder', type=json.loads, default={}, help='encoder architecture')
    parser.add_argument('-masks', nargs='+', default=[], help='marginalzation masks')
    parser.add_argument('-filter', default=None, help='mask filter')
    parser.add_argument('-arbitrary', default=False, action='store_true', help='arbitrary design')
    parser.add_argument('-weights', default=None, help='warm-start weights')

    parser.add_argument('-criterion', default='NLL', choices=['NLL', 'FL', 'PL', 'QS'], help='optimization criterion')
    parser.add_argument('-adversary', default=None, help='adversary network file (PTH)')
    parser.add_argument('-distillation', default=None, help='distillation network file (PTH)')
    parser.add_argument('-lambdas', type=float, nargs=2, default=(1e-3, 1e-3), help='auxiliary losses weights')

    parser.add_argument('-epochs', type=int, default=256, help='number of epochs')
    parser.add_argument('-per-epoch', type=int, default=256, help='batches per epoch')
    parser.add_argument('-bs', type=int, default=1024, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weight-decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-amsgrad', type=bool, default=False, help='AMS gradient')
    parser.add_argument('-patience', type=int, default=5, help='scheduler patience')
    parser.add_argument('-threshold', type=float, default=1e-2, help='scheduler threshold')
    parser.add_argument('-factor', type=float, default=5e-1, help='scheduler factor')
    parser.add_argument('-min-lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('-clip', type=float, default=1e2, help='gradient norm')

    parser.add_argument('-valid', default=None, help='validation samples file (H5)')

    parser.add_argument('-o', '--output', default='products/models/out.pth', help='output file (PTH)')

    args = parser.parse_args()
    args.date = datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')

    # Output directory
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Simulator & Model
    settings = vars(args)
    simulator, dataset, model = build_instance(settings)

    ## Arbitrary masks
    if args.arbitrary:
        theta_size = simulator.prior.sample().numel()

        if not args.masks:
            args.masks.append('uniform')

        if args.masks[0] == 'poisson':
            filtr = None if args.filter is None else amnre.str2mask(args.filter)
            mask_sampler = amnre.PoissonMask(theta_size, filtr=filtr)
        elif args.masks[0] == 'uniform':
            mask_sampler = amnre.UniformMask(theta_size)
        else:
            masks = amnre.list2masks(args.masks, theta_size, args.filter)
            mask_sampler = amnre.SelectionMask(masks)

        mask_sampler.to(args.device)

    # Criterion(s)
    if args.criterion == 'FL':
        criterion = amnre.FocalWithLogitsLoss()
    elif args.criterion == 'PL':
        criterion = amnre.PeripheralWithLogitsLoss()
    elif args.criterion == 'QS':
        criterion = amnre.QSWithLogitsLoss()
    else:  # args.criterion == 'NLL':
        criterion = amnre.NLLWithLogitsLoss()

    ## Adversarial
    if args.adversary is None:
        adversary = Dummy()
    elif os.path.isfile(args.adversary):
        adversary = load_model(args.adversary)
        adversary.to(args.device)
        adversary.eval()
    else:
        adversary = Dummy()

    for p in adversary.parameters():
        p.requires_grad = False

    ## Distillation
    if args.distillation is None:
        targetnet = None
    elif os.path.isfile(args.distillation):
        targetnet = load_model(args.distillation)
        targetnet.to(args.device)
        targetnet.eval()
    elif simulator.tractable:
        targetnet = amnre.LTERatio(simulator)
    else:
        targetnet = None

    rr_loss = amnre.RRLoss()
    sr_loss = amnre.SRLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    scheduler = amnre.ReduceLROnPlateau(
        optimizer,
        factor=args.factor,
        patience=args.patience,
        threshold=args.threshold,
        min_lr=args.min_lr,
    )

    # Datasets
    trainset = amnre.LTEDataset(dataset)

    if args.valid is not None:
        validset = amnre.LTEDataset(amnre.OfflineDataset(args.valid, batch_size=args.bs, device=args.device))

    # Routine
    def routine(dataset, optimize: bool = True) -> Tuple[float, torch.Tensor]:
        losses = []

        start = time()

        for theta, theta_prime, x in islice(dataset, args.per_epoch):
            theta.requires_grad = True
            z = model.encoder(x)
            adv_z = adversary.encoder(x)

            if args.arbitrary:
                if model.hyper is None:
                    mask = mask_sampler(theta.shape[:1])
                else:
                    mask = mask_sampler()
                    model[mask]
                    adversary[mask]
                    mask = None

                ratio = model(theta, z, mask)
                ratio_prime = model(theta_prime, z, mask)

                adv_ratio_prime = adversary(theta_prime, adv_z, mask)
            else:
                ratio = model(theta, z)
                ratio_prime = model(theta_prime, z)

                adv_ratio_prime = adversary(theta_prime, adv_z)

            if adv_ratio_prime is not None:
                adv_ratio_prime = adv_ratio_prime.exp()

            l = criterion(ratio) + criterion(-ratio_prime, adv_ratio_prime)

            if targetnet is not None:
                target_z = targetnet.encoder(x)
                target_ratio = targetnet(theta, target_z)
                target_ratio_prime = targetnet(theta_prime, target_z)

                l_rr = args.lambdas[0] * rr_loss(ratio_prime, target_ratio_prime)
                l_irr = args.lambdas[0] * rr_loss(-ratio, -target_ratio)
                l_sr = args.lambdas[1] * sr_loss(theta, ratio, target_ratio)

                l = torch.stack([l, l_rd, l_ird, l_sd])

            losses.append(l.tolist())

            if optimize:
                optimizer.zero_grad()
                l.sum().backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

        end = time()

        return end - start, torch.tensor(losses)

    # Training
    stats = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        timing, losses = routine(trainset)

        stats.append({
            'epoch': epoch,
            'time': timing,
            'lr': scheduler.lr,
            'mean': losses.mean(dim=0).tolist(),
            'std': losses.std(dim=0).tolist(),
        })

        if args.valid is not None:
            model.eval()

            with torch.no_grad() if targetnet is None else nullcontext():
                _, v_losses = routine(validset, False)

            model.train()

            stats[-1].update({
                'v_mean': v_losses.mean(dim=0).tolist(),
                'v_std': v_losses.std(dim=0).tolist(),
            })

            scheduler.step(v_losses.mean())
        else:
            scheduler.step(losses.mean())

        if scheduler.plateau:
            df = pd.DataFrame(stats)
            df.to_csv(args.output.replace('.pth', '.csv'), index=False)

            if scheduler.bottom:
                break

    # Outputs

    ## Weights
    if hasattr(model, 'clear'):
        model.clear()

    torch.save(model.cpu().state_dict(), args.output)

    ## Settings
    with open(args.output.replace('.pth', '.json'), 'w') as f:
        json.dump(settings, f, indent=4)
