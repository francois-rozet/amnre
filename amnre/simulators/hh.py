#!/usr/bin/env python

import numpy as np
import torch

from functools import cached_property
from multiprocessing import cpu_count
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import Tuple, List

from . import Simulator


class HH(Simulator):
    r"""Hodgkin-Huxley

    References:
        https://github.com/mackelab/IdentifyMechanisticModels_2020/tree/master/5_hh
    """

    def __init__(self,
        seed: int = 0,
        cython: bool = True,
        n_xcorr: int = 0,
        n_mom: int = 4,
        n_summary: int = 7,
    ):
        super().__init__()

        from delfi.generator import MPGenerator

        from .hhpkg.utils import obs_params, syn_current, syn_obs_data, syn_obs_stats, prior
        from .hhpkg.HodgkinHuxley import HodgkinHuxley
        from .hhpkg.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments

        if cython:
            try:
                from .hhpkg import biophys_cython_comp
            except ImportError:
                cython = False

        theta, _ = obs_params(reduced_model=False)
        I, t_on, t_off, dt = syn_current()
        observation = syn_obs_data(I, dt, theta, seed=seed, cython=cython)

        prior = prior(
            true_params=theta,
            prior_uniform=True,
            prior_extent=True,
            prior_log=False,
            seed=seed,
        )

        models = [
            HodgkinHuxley(
                I, dt, V0=observation['data'][0],
                reduced_model=False,
                prior_log=False,
                seed=seed + i,
                cython=cython,
            ) for i in range(cpu_count())
        ]

        stats = HodgkinHuxleyStatsMoments(t_on=t_on, t_off=t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)

        self.generator = MPGenerator(models=models, prior=prior, summary=stats)

        low = torch.from_numpy(prior.lower).float()
        high = torch.from_numpy(prior.upper).float()

        self.register_buffer('low', low)
        self.register_buffer('high', high)

        self.theta_star = theta[None]
        self.x_star = syn_obs_stats(
            data=observation,
            I=I, t_on=t_on, t_off=t_off, dt=dt,
            params=theta,
            seed=seed, cython=cython, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary,
        )

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return Independent(Uniform(self.low[mask], self.high[mask]), 1)

    @cached_property
    def labels(self) -> List[str]:
        labels = [r'g_{\mathrm{Na}}', r'g_{\mathrm{K}}', 'g_l', r'g_{\mathrm{M}}', r'\tau_{\max}', '-V_T', r'\sigma', '-E_l']
        labels = [f'${l}$' for l in labels]

        return labels

    def joint(self, shape: torch.Size = ()) -> Tuple[np.ndarray, np.ndarray]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        numel = torch.Size(shape).numel()
        theta, x = self.generator.gen(numel, verbose=False)

        theta = theta.reshape(shape + theta.shape[1:]).astype(np.float32)
        x = x.reshape(shape + x.shape[1:]).astype(np.float32)

        return theta, x

    def events(self) -> Tuple[np.ndarray, np.ndarray]:
        r""" (theta*, x*) """

        return self.theta_star.astype(np.float32), self.x_star.astype(np.float32)
