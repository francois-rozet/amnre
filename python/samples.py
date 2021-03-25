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

    parser.add_argument('-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-simulator', default='SLCP', choices=['SLCP', 'MLCP', 'GW'])

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-n', type=int, default=2 ** 20, help='number of samples')
    parser.add_argument('-chunk-size', type=int, default=2 ** 14, help='chunk size')
    parser.add_argument('-batch-size', type=int, default=2 ** 10, help='batch size')

    parser.add_argument('-o', '--output', default='../products/samples/out.h5', help='output file (H5)')

    args = parser.parse_args()

    # Simulator
    if args.simulator == 'GW':
        simulator = amsi.GW()
    elif args.simulator == 'MLCP':
        simulator = amsi.MLCP()
    else:  # args.simulator == 'SCLP'
        simulator = amsi.SLCP()

    simulator.to(args.device)

    # Placeholders
    theta, x = simulator.sample()
    theta_shape, x_shape = theta.shape, x.shape

    # Seed
    torch.manual_seed(args.seed)

    # Fill dataset
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, 'w') as f:

        theta_set = f.create_dataset(
            'theta',
            (args.n,) + theta_shape,
            chunks=(args.chunk_size,) + theta_shape,
            maxshape=(None,) + theta_shape,
            dtype='float32',
        )

        x_set = f.create_dataset(
            'x',
            (args.n,) + x_shape,
            chunks=(args.chunk_size,) + x_shape,
            maxshape=(None,) + x_shape,
            dtype='float32',
        )

        ## Sampling
        with tqdm(total=args.n) as tq:
            for i in range(0, args.n, args.chunk_size):
                theta_chunk, x_chunk = [], []

                for _ in range(0, args.chunk_size, args.batch_size):
                    theta, x = simulator.sample((args.batch_size,))
                    theta, x = theta.cpu().numpy(), x.cpu().numpy()

                    theta_chunk.append(theta)
                    x_chunk.append(x)

                    tq.update(args.batch_size)

                theta_set[i:i + args.chunk_size] = np.concatenate(theta_chunk)
                x_set[i:i + args.chunk_size] = np.concatenate(x_chunk)
