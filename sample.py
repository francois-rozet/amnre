#!/usr/bin/env python

import h5py
import numpy as np
import os
import torch
import torch.utils.data as data

from tqdm import tqdm

import amnre

from amnre.simulators.slcp import SLCP
from amnre.simulators.gw import GW
from amnre.simulators.hh import HH


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sampling')

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'GW', 'HH'])
    parser.add_argument('-live', default=False, action='store_true', help='live samples')  # only GW

    parser.add_argument('-seed', type=int, default=None, help='random seed')
    parser.add_argument('-samples', type=int, default=2 ** 20, help='number of samples')
    parser.add_argument('-chunk-size', type=int, default=2 ** 16, help='chunk size')
    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')

    parser.add_argument('-reference', default=None, help='dataset of reference (H5)')
    parser.add_argument('-events', default=False, action='store_true', help='store events')

    parser.add_argument('-moments', default=False, action='store_true', help='compute moments')
    parser.add_argument('-dump', type=int, default=2 ** 16, help='number of dump samples')

    parser.add_argument('-o', '--output', default='products/samples/out.h5', help='output file (H5)')

    args = parser.parse_args()

    args.chunk_size = min(args.chunk_size, args.samples)
    args.batch_size = min(args.batch_size, args.chunk_size)

    # Seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Simulator
    if args.simulator == 'GW':
        simulator = GW(noisy=not args.live)
    elif args.simulator == 'HH':
        simulator = HH()
    else:  # args.simulator == 'SCLP'
        simulator = SLCP()

    # Moments
    if args.moments:
        if args.reference is None:
            x = [
                x for _, x in
                tqdm(
                    amnre.ParallelSampler(simulator, args.dump, args.batch_size),
                    unit_scale=args.batch_size,
                )
            ]
            x = np.concatenate(x)

            mu = x.mean(axis=0)
            sigma = x.std(axis=0)
        else:
            with h5py.File(args.reference) as f:
                mu = f['mu'][:]
                sigma = f['sigma'][:]

    # Fill dataset
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, 'w') as f:
        ## Moments
        if args.moments:
            f.create_dataset('mu', data=mu)
            f.create_dataset('sigma', data=sigma)

        ## Events
        if args.events:
            theta, x = simulator.events

            if theta is not None:
                f.create_dataset('theta', data=np.asarray(theta))
            f.create_dataset('x', data=np.asarray(x))

            exit()

        ## Samples
        theta, x = simulator.joint()
        theta, x = np.asarray(theta), np.asarray(x)

        f.create_dataset(
            'theta',
            (args.samples,) + theta.shape,
            chunks=(args.chunk_size,) + theta.shape,
            dtype=theta.dtype,
        )

        f.create_dataset(
            'x',
            (args.samples,) + x.shape,
            chunks=(args.chunk_size,) + x.shape,
            dtype=x.dtype,
        )

        i = 0

        for theta, x in tqdm(
            amnre.ParallelSampler(simulator, args.samples, args.batch_size),
            unit_scale=args.batch_size,
        ):
            j = i + args.batch_size
            f['theta'][i:j] = np.asarray(theta)
            f['x'][i:j] = np.asarray(x)
            i = j
