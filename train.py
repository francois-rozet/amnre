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
from tqdm import tqdm
from typing import List, Tuple

import amnre

from amnre.simulators.slcp import SLCP
from amnre.simulators.gw import GW
from amnre.simulators.hh import HH


def build_embedding(input_size: torch.Size, name: str = None, **kwargs) -> tuple:
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
    live = None

    if settings['simulator'] == 'GW':
        simulator = GW()

        if settings.get('live', True):
            live = simulator.noise
    elif settings['simulator'] == 'HH':
        simulator = HH()
    else:  # settings['simulator'] == 'SCLP'
        simulator = SLCP()

    simulator.to(settings['device'])

    # Dataset
    if settings['samples'] is None:
        dataset = amnre.OnlineDataset(simulator, batch_size=settings['bs'])
        theta, x = simulator.joint()
    else:
        dataset = amnre.OfflineDataset(settings['samples'], batch_size=settings['bs'], device=settings['device'], live=live)
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

    # Model & embedding
    embedding, x_size = build_embedding(x.shape, **settings['embedding'])

    model_args = settings['model'].copy()
    model_args['embedding'] = embedding
    model_args['moments'] = moments

    if settings['arbitrary']:
        model_args['hyper'] = settings['hyper']
        model = amnre.AMNRE(theta_size, x_size, **model_args)
    else:
        masks = amnre.list2masks(settings['masks'], theta_size, settings['filter'])

        if len(masks) == 0:
            if settings['flow']:
                model = amnre.NPE(theta_size, x_size, prior=simulator.prior, **model_args)
            else:
                model = amnre.NRE(theta_size, x_size, **model_args)
        else:
            if settings['flow']:
                model = amnre.MNPE(masks, x_size, priors=[simulator.masked_prior(m) for m in masks], **model_args)
            else:
                model = amnre.MNRE(masks, x_size, **model_args)

    ## Weights
    if settings['weights'] is not None:
        weights = torch.load(settings['weights'], map_location='cpu')
        model.load_state_dict(weights)

    model.to(settings['device'])

    # Adversary
    if os.path.isfile(settings['adversary']) and type(model) in [amnre.NRE, amnre.MNRE]:
        adversary = load_model(settings['adversary'])
        adversary.to(settings['device'])
        adversary.eval()

        if type(adversary) in [amnre.NPE, amnre.MNPE]:
            adversary.ratio()

        if type(model) is amnre.MNRE:
            if type(adversary) in [amnre.MNRE, amnre.MNPE]:
                adversary.filter(model.masks)
            elif type(adversary) in [amnre.AMNRE]:
                adversary[model.masks]
    else:
        adversary = amnre.Dummy()

    for p in adversary.parameters():
        p.requires_grad = False

    return simulator, dataset, model, adversary


def load_settings(filename: str) -> dict:
    with open(filename) as f:
        settings = json.load(f)

    return settings


def load_model(filename: str) -> nn.Module:
    settings = load_settings(filename.replace('.pth', '.json'))
    settings['weights'] = filename

    _, _, model, _ = build_instance(settings)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'GW', 'HH'])
    parser.add_argument('-samples', default=None, help='samples file (H5)')
    parser.add_argument('-live', default=False, action='store_true', help='live samples')  # only GW

    parser.add_argument('-model', type=json.loads, default={}, help='model architecture')
    parser.add_argument('-hyper', type=json.loads, default=None, help='hypernet architecture')
    parser.add_argument('-embedding', type=json.loads, default={}, help='embedding architecture')
    parser.add_argument('-flow', default=False, action='store_true', help='normalizing flow')
    parser.add_argument('-arbitrary', default=False, action='store_true', help='arbitrary architecture')
    parser.add_argument('-masks', nargs='+', default=[], help='marginalzation masks')
    parser.add_argument('-filter', default=None, help='mask filter')
    parser.add_argument('-weights', default=None, help='warm-start weights')

    parser.add_argument('-criterion', default='NLL', choices=['NLL', 'FL', 'PL', 'QS'], help='optimization criterion')
    parser.add_argument('-adversary', default='notafile.pth', help='adversary network file (PTH)')
    parser.add_argument('-inverse', default=False, action='store_true', help='inverse adversary')

    parser.add_argument('-epochs', type=int, default=256, help='number of epochs')
    parser.add_argument('-descents', type=int, default=256, help='descents per epoch')
    parser.add_argument('-bs', type=int, default=1024, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weight-decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-amsgrad', type=bool, default=False, help='AMS gradient')
    parser.add_argument('-patience', type=int, default=7, help='scheduler patience')
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
    simulator, dataset, model, adversary = build_instance(settings)

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
    else:
        mask_sampler = None

    # Criterion(s)
    if args.flow:
        criterion = amnre.NLL()
    elif args.criterion == 'FL':
        criterion = amnre.FocalWithLogitsLoss()
    elif args.criterion == 'PL':
        criterion = amnre.PeripheralWithLogitsLoss()
    elif args.criterion == 'QS':
        criterion = amnre.QSWithLogitsLoss()
    else:  # args.criterion == 'NLL':
        criterion = amnre.NLLWithLogitsLoss()

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
        validset = amnre.OfflineDataset(args.valid, batch_size=args.bs, device=args.device)
        validset = amnre.LTEDataset(validset)

    # Training
    stats = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        duration, losses = amnre.routine(
            model,
            trainset,
            criterion,
            optimizer=optimizer,
            adversary=adversary,
            inverse=args.inverse,
            descents=args.descents,
            flow=args.flow,
            mask_sampler=mask_sampler,
            clip=args.clip,
        )

        stats.append({
            'epoch': epoch,
            'time': duration,
            'lr': scheduler.lr,
            'mean': losses.mean(dim=0).tolist(),
            'std': losses.std(dim=0).tolist(),
        })

        if args.valid is not None:
            with torch.no_grad():
                model.eval()
                _, v_losses = amnre.routine(
                    model,
                    validset,
                    criterion,
                    optimizer=None,
                    adversary=adversary,
                    inverse=args.inverse,
                    flow=args.flow,
                    mask_sampler=mask_sampler,
                )

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
