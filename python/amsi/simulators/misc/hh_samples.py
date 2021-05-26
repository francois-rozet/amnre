#!/usr/bin/env python

import h5py
import multiprocessing as mp
import numpy as np
import os

from tqdm import tqdm
from typing import Tuple

from delfi.generator import MPGenerator

from model import utils
from model.HodgkinHuxley import HodgkinHuxley
from model.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments


def numel(shape: Tuple[int, ...]) -> int:
    prod = 1

    for s in shape:
        prod *= s

    return prod


class HH:
    def __init__(
        self,
        seed: int = 0,
        cython: bool = True,
        n_xcorr: int = 0,
        n_mom: int = 4,
        n_summary: int = 7,
        summary_stats: int = 1,
    ):
        theta, _ = utils.obs_params(reduced_model=False)
        I, t_on, t_off, dt = utils.syn_current()
        obs = utils.syn_obs_data(I, dt, theta, seed=seed, cython=cython)

        self._theta = theta[None]
        self._x = utils.syn_obs_stats(data=obs, I=I, t_on=t_on, t_off=t_off, dt=dt, params=theta, seed=seed, cython=cython, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary, summary_stats=summary_stats)

        self.prior = utils.prior(
            true_params=theta,
            prior_uniform=True,
            prior_extent=True,
            prior_log=False,
            seed=seed,
        )

        self.models = [
            HodgkinHuxley(
                I, dt, V0=obs['data'][0],
                reduced_model=False,
                prior_log=False,
                seed=seed + i,
                cython=cython,
            ) for i in range(mp.cpu_count())
        ]

        self.stats = HodgkinHuxleyStatsMoments(t_on=t_on, t_off=t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)

        self.generator = MPGenerator(models=self.models, prior=self.prior, summary=self.stats)

    def sample(self, shape: Tuple[int, ...] = ()) -> Tuple[np.ndarray, np.ndarray]:
        theta, x = self.generator.gen(numel(shape), verbose=False)

        theta = theta.astype(np.float32).reshape(shape + theta.shape[1:])
        x = x.astype(np.float32).reshape(shape + x.shape[1:])

        return theta, x

    def events(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._theta.astype(np.float32), self._x.astype(np.float32)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Samples')

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-samples', type=int, default=2 ** 20, help='number of samples')
    parser.add_argument('-chunk-size', type=int, default=2 ** 16, help='chunk size')
    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')

    parser.add_argument('-events', default=False, action='store_true', help='events dataset')

    parser.add_argument('-o', '--output', default='out.h5', help='output file (H5)')

    args = parser.parse_args()

    args.chunk_size = min(args.chunk_size, args.samples)
    args.batch_size = min(args.batch_size, args.chunk_size)

    # Simulator
    simulator = HH(seed=args.seed)

    # Fill dataset
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, 'w') as f:
        ## Events
        if args.events:
            theta, x = simulator.events()

            f.create_dataset('theta', data=theta)
            f.create_dataset('x', data=x)

            exit()

        ## Samples
        theta, x = simulator.sample()

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

        with tqdm(total=args.samples) as tq:
            for i in range(0, args.samples, args.chunk_size):
                theta_chunk, x_chunk = [], []

                for _ in range(0, args.chunk_size, args.batch_size):
                    theta, x = simulator.sample((args.batch_size,))

                    theta_chunk.append(theta)
                    x_chunk.append(x)

                    tq.update(args.batch_size)

                theta_set[i:i + args.chunk_size] = np.concatenate(theta_chunk)
                x_set[i:i + args.chunk_size] = np.concatenate(x_chunk)
