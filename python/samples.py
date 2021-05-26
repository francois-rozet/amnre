#!/usr/bin/env python

import h5py
import numpy as np
import os
import torch

from tqdm import tqdm

import amsi


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Samples')

    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW'])

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-samples', type=int, default=2 ** 20, help='number of samples')
    parser.add_argument('-chunk-size', type=int, default=2 ** 16, help='chunk size')
    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')

    parser.add_argument('-basis-size', type=int, default=128, help='basis size')
    parser.add_argument('-basis-samples', type=int, default=2 ** 15, help='number of samples for the basis')
    parser.add_argument('-basis-from', default=None, help='basis file (H5)')

    parser.add_argument('-events', default=False, action='store_true', help='events dataset')

    parser.add_argument('-o', '--output', default='../products/samples/out.h5', help='output file (H5)')

    args = parser.parse_args()

    args.chunk_size = min(args.chunk_size, args.samples)
    args.batch_size = min(args.batch_size, args.chunk_size)

    # Seed
    torch.manual_seed(args.seed)

    # Simulator
    if args.simulator == 'GW':
        if args.basis_from is None:
            simulator = amsi.BasisGW(args.basis_size, args.basis_samples)
        else:
            with h5py.File(args.basis_from, 'r') as f:
                simulator = amsi.GW(basis=f['basis'][:])
    elif args.simulator == 'MLCP':
        simulator = amsi.MLCP()
    else:  # args.simulator == 'SCLP'
        simulator = amsi.SLCP()

    # Fill dataset
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, 'w') as f:
        ## Basis
        if args.simulator == 'GW':
            f.create_dataset('basis', data=simulator.basis)

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
        else:
            noise_set = None

        with tqdm(total=args.samples) as tq:
            for i in range(0, args.samples, args.chunk_size):
                theta_chunk, x_chunk, noise_chunk = [], [], []

                for _ in range(0, args.chunk_size, args.batch_size):
                    theta, x = simulator.sample((args.batch_size,))
                    theta, x = np.asarray(theta), np.asarray(x)

                    theta_chunk.append(theta)
                    x_chunk.append(x)

                    if noise_set is not None:
                        noise = simulator.noise((args.batch_size,))
                        noise = np.asarray(noise)
                        noise_chunk.append(noise)

                    tq.update(args.batch_size)

                theta_set[i:i + args.chunk_size] = np.concatenate(theta_chunk)
                x_set[i:i + args.chunk_size] = np.concatenate(x_chunk)

                if noise_set is not None:
                    noise_set[i:i + args.chunk_size] = np.concatenate(noise_chunk)
