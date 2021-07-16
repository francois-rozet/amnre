#!/usr/bin/env python

import h5py
import numpy as np
import os
import torch

from tqdm import tqdm

import amnre


def gather_chunk(
    simulator: amnre.Simulator,
    chunk_size: int,
    batch_size: int,
    noisy: bool = True,
    progress = None  # tqdm
) -> tuple:
    noisy = noisy and hasattr(simulator, 'noise')
    theta_chunk, x_chunk, noise_chunk = [], [], []

    for _ in range(0, chunk_size, batch_size):
        theta, x = simulator.sample((batch_size,))
        theta, x = np.asarray(theta), np.asarray(x)

        theta_chunk.append(theta)
        x_chunk.append(x)

        if noisy:
            noise = simulator.noise((batch_size,))
            noise = np.asarray(noise)
            noise_chunk.append(noise)

        if progress is not None:
            progress.update(batch_size)

    theta_chunk = np.concatenate(theta_chunk)
    x_chunk = np.concatenate(x_chunk)
    noise_chunk = np.concatenate(noise_chunk) if noisy else None

    return theta_chunk, x_chunk, noise_chunk


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample from simulator')

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW', 'HH'])

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-samples', type=int, default=2 ** 20, help='number of samples')
    parser.add_argument('-chunk-size', type=int, default=2 ** 16, help='chunk size')
    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')

    parser.add_argument('-reference', default=None, help='dataset of reference (H5)')
    parser.add_argument('-events', default=False, action='store_true', help='store events')

    parser.add_argument('-moments', default=False, action='store_true', help='compute moments')

    parser.add_argument('-o', '--output', default='products/samples/out.h5', help='output file (H5)')

    args = parser.parse_args()

    args.chunk_size = min(args.chunk_size, args.samples)
    args.batch_size = min(args.batch_size, args.chunk_size)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Simulator
    if args.simulator == 'GW':
        simulator = amnre.GW()
    elif args.simulator == 'HH':
        simulator = amnre.HH(seed=args.seed)
    elif args.simulator == 'MLCP':
        simulator = amnre.MLCP()
    else:  # args.simulator == 'SCLP'
        simulator = amnre.SLCP()

    # Moments
    if args.moments:
        if args.reference is None:
            with tqdm(total=args.chunk_size) as tq:
                _, x, _ = gather_chunk(simulator, args.chunk_size, args.batch_size, progress=tq)

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
            theta, x = simulator.events()

            if theta is not None:
                f.create_dataset('theta', data=np.asarray(theta))
            f.create_dataset('x', data=np.asarray(x))

            exit()

        ## Samples
        theta, x = simulator.sample()
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

        if hasattr(simulator, 'noise'):
            f.create_dataset_like('noise', f['x'])

        with tqdm(total=args.samples) as tq:
            for i in range(0, args.samples, args.chunk_size):
                theta, x, noise = gather_chunk(
                    simulator,
                    args.chunk_size,
                    args.batch_size,
                    progress=tq,
                )

                f['theta'][i:i + args.chunk_size] = theta
                f['x'][i:i + args.chunk_size] = x

                if noise is not None:
                    f['noise'][i:i + args.chunk_size] = noise
