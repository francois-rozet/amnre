#!/usr/bin/env python

import h5py
import numpy as np
import os
import torch

from tqdm import tqdm

import amsi


def gather_chunk(
    simulator: amsi.Simulator,
    chunk_size: int,
    batch_size: int,
    noisy: bool = True,
    progress = None
) -> tuple:
    noisy = noisy and hasattr(simulator, 'noise')
    theta_chunk, x_chunk, noise_chunk = [], [], []

    if progress is None:
        progress = tqdm(total=args.chunk_size)

    for _ in range(0, chunk_size, batch_size):
        theta, x = simulator.sample((batch_size,))
        theta, x = np.asarray(theta), np.asarray(x)

        theta_chunk.append(theta)
        x_chunk.append(x)

        if noisy:
            noise = simulator.noise((batch_size,))
            noise = np.asarray(noise)
            noise_chunk.append(noise)

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

    parser.add_argument('-o', '--output', default='../products/samples/out.h5', help='output file (H5)')

    args = parser.parse_args()

    args.chunk_size = min(args.chunk_size, args.samples)
    args.batch_size = min(args.batch_size, args.chunk_size)

    # Seed
    torch.manual_seed(args.seed)

    # Simulator
    if args.simulator == 'GW':
        if args.reference is None:
            simulator = amsi.GW(fiducial=True)

            _, x, _ = gather_chunk(simulator, args.chunk_size, args.batch_size, noisy=False)
            x = x.reshape((-1, x.shape[-1]))

            simulator.fiducial = False
            simulator.basis = amsi.svd_basis(x)
        else:
            with h5py.File(args.reference) as f:
                simulator = amsi.GW(basis=f['basis'][:])
    elif args.simulator == 'HH':
        if args.reference is None:
            simulator = amsi.HH()

            _, x, _ = gather_chunk(simulator, args.chunk_size, args.batch_size, noisy=False)

            simulator.mu = x.mean(axis=0)
            simulator.sigma = x.std(axis=0)
        else:
            with h5py.File(args.reference) as f:
                simulator = amsi.HH(f['mu'][:], f['sigma'][:])
    elif args.simulator == 'MLCP':
        simulator = amsi.MLCP()
    else:  # args.simulator == 'SCLP'
        simulator = amsi.SLCP()

    # Fill dataset
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, 'w') as f:
        ## Preprocessing
        if args.simulator == 'GW':
            f.create_dataset('basis', data=simulator.basis)
        elif args.simulator == 'HH':
            f.create_dataset('mu', data=simulator.mu)
            f.create_dataset('sigma', data=simulator.sigma)

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

        theta_set = f.create_dataset(
            'theta',
            (args.samples,) + theta.shape,
            chunks=(args.chunk_size,) + theta.shape,
            dtype=theta.dtype,
        )

        x_set = f.create_dataset(
            'x',
            (args.samples,) + x.shape,
            chunks=(args.chunk_size,) + x.shape,
            dtype=x.dtype,
        )

        if hasattr(simulator, 'noise'):
            noise_set = f.create_dataset_like('noise', x_set)

        with tqdm(total=args.samples) as tq:
            for i in range(0, args.samples, args.chunk_size):
                theta_chunk, x_chunk, noise_chunk = gather_chunk(
                    simulator,
                    args.chunk_size,
                    args.batch_size,
                    progress=tq,
                )

                theta_set[i:i + args.chunk_size] = theta_chunk
                x_set[i:i + args.chunk_size] = x_chunk

                if noise_chunk is not None:
                    noise_set[i:i + args.chunk_size] = noise_chunk
