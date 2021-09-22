#!/usr/bin/env python

import h5py
import inspect
import math
import numpy as np
import os
import torch
import torch.nn as nn

from functools import cached_property
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import List, Tuple

from . import Simulator


class GW(Simulator):
    r"""Gravitational Waves

    References:
        https://github.com/stephengreen/lfi-gw
    """

    def __init__(
        self,
        n_rb: int = 2 ** 7,  # 128
        n_ref: int = 2 ** 15,  # 32768
        reduced_basis: bool = True,
        noisy: bool = True,
    ):
        super().__init__()

        # Prior
        bounds = torch.tensor([
            [10., 80.],  # primary mass [solar masses]
            [10., 80.],  # secondary mass [solar masses]
            [0., 2 * math.pi],  # coalesence phase [rad]
            [-0.1, 0.1],  # coalescence time [s]
            [100., 1000.],  # luminosity distance [megaparsec]
            [0., 0.88],  # a_1 [/]
            [0., 0.88],  # a_2 [/]
            [0., math.pi],  # tilt_1 [rad]
            [0., math.pi],  # tilt_2 [rad]
            [0., 2 * math.pi],  # phi_12 [rad]
            [0., 2 * math.pi],  # phi_jl [rad]
            [0., math.pi],  # theta_jn [rad]
            [0., math.pi],  # polarization [rad]
            [0., 2 * math.pi],  # right ascension [rad]
            [-math.pi / 2, math.pi / 2],  # declination [rad]
        ])

        self.register_buffer('low', bounds[:, 0])
        self.register_buffer('high', bounds[:, 1])

        self.reduced_basis = reduced_basis
        self.noisy = noisy

        # Simulator
        from .lfigw import waveform_generator as wfg
        from .lfigw.reduced_basis import SVDBasis

        path = os.path.dirname(inspect.getfile(wfg))
        path = os.path.join(path, 'events/GW150914/')

        wfd = wfg.WaveformDataset(spins_aligned=False, domain='RB', extrinsic_at_train=True)

        wfd.Nrb = n_rb
        wfd.approximant = 'IMRPhenomPv2'
        wfd.prior['distance'] = [100.0, 1000.0]
        wfd.prior['a_1'][1] = 0.88
        wfd.prior['a_2'][1] = 0.88
        wfd.distance_prior_fn = 'uniform_distance'

        self.wfd = wfd

        ## Event
        wfd.load_event(path)

        with h5py.File(os.path.join(path, 'strain_FD_whitened.hdf5')) as f:
            self.x_star = np.stack([f['H1'][:], f['L1'][:]]).astype(np.complex64)

        ## Basis
        if self.reduced_basis:
            try:
                wfd.basis = SVDBasis()
                wfd.basis.load(path)
                assert wfd.basis.n >= n_rb
                wfd.basis.truncate(n_rb)
            except:
                wfd.generate_reduced_basis(n_ref, n_ref)
                wfd.basis.save(path)

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        if mask is ...:
            mask = [True] * len(self.low)

        marginals = []

        if mask[0] or mask[1]:
            if mask[0] and mask[1]:
                law = SortUniform
            elif mask[0]:
                law = MaxUniform
            elif mask[1]:
                law = MinUniform

            marginals.append(law(self.low[0], self.high[0]))

        for i, b in enumerate(mask[2:], start=2):
            if not b:
                continue

            if i in [7, 8, 11]:  # [tilt_1, tilt_2, theta_jn]
                m = SinAngle(self.low[i], self.high[i])
            elif i == 14:  # declination
                m = CosAngle(self.low[i], self.high[i])
            else:
                m = Uniform(self.low[i], self.high[i])

            marginals.append(m)

        return Joint(marginals)

    @cached_property
    def labels(self) -> List[str]:
        labels = [
            'm_1', 'm_2', r'\phi_c', 't_c', 'd_L',
            'a_1', 'a_2', r'\theta_1', r'\theta_2', r'\phi_{12}', r'\phi_{JL}',
            r'\theta_{JN}', r'\psi', r'\alpha', r'\delta',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def sample(self, theta: torch.Tensor, shape: torch.Size = ()) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        p = theta.cpu().numpy().astype(np.float64)
        x = np.vectorize(self._simulate, signature='(m)->(n,p)')(p)
        x = self.postprocess(x)

        if shape:
            x = np.broadcast_to(x, shape + x.shape)

        x = torch.from_numpy(x).to(theta.device)
        x = torch.view_as_real(x)

        if self.noisy:
            x = self.noise(theta, x)

        return x

    def _simulate(self, theta: np.ndarray) -> np.ndarray:
        x = self.wfd._generate_whitened_waveform(theta)
        x = list(x.values())

        return np.stack(x).astype(np.complex64)

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        if self.reduced_basis:
            x = self.wfd.basis.fseries_to_basis_coefficients(x)

        return x / self.wfd._noise_std

    def noise(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x)

    @property
    def events(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" x* """

        x = self.postprocess(self.x_star)
        x = torch.tensor(x)
        x = torch.view_as_real(x)

        return None, x[None]


class SortUniform(Uniform):
    r"""TODO"""

    def __init__(self, low: torch.Tensor, high: torch.Tensor, n: int = 2):
        super().__init__(low, high)

        self.n = n
        self.log_coef = math.log(math.factorial(n))

    @property
    def event_shape(self) -> torch.Size:
        return super().event_shape + (self.n,)

    def sample(self, shape: torch.Size = ()) -> torch.Tensor:
        value = super().sample(shape + (self.n,))
        value, _ = torch.sort(value, descending=True)

        return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        sorted = (value[..., :-1] >= value[..., 1:]).all(dim=-1)
        return self.log_coef + sorted.log() + super().log_prob(value).sum(dim=-1)


class MaxUniform(Uniform):
    r"""TODO"""

    def __init__(self, low: torch.Tensor, high: torch.Tensor, n: int = 2):
        super().__init__(low, high)

        self.n = n
        self.log_volume = n * (self.high - self.low).log() - math.log(n)

    def sample(self, shape: torch.Size = ()) -> torch.Tensor:
        value = super().sample(shape + (self.n,))
        value, _ = torch.max(value, dim=-1)

        return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return (self.n - 1) * (value - self.low).log() - self.log_volume


class MinUniform(MaxUniform):
    r"""TODO"""

    def sample(self, shape: torch.Size = ()) -> torch.Tensor:
        value = super().sample(shape + (self.n,))
        value, _ = torch.min(value, dim=-1)

        return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return (self.n - 1) * (self.high - value).log() - self.log_volume


class SuperUniform(Uniform):
    r"""Abstract more-than-uniform distribution"""

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__(self._func(low), self._func(high))

        self.log_volume = (self.high - self.low).log()
        self.low_, self.high_ = low, high

    def sample(self, shape: torch.Size = ()) -> torch.Tensor:
        return self._ifunc(super().sample(shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        mask = torch.logical_and(value.ge(self.low_), value.le(self.high_))
        return (mask * self._dfunc(value).relu()).log() - self.log_volume


class PowerLaw(SuperUniform):
    r"""Power law distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.power_law.UniformPowerLaw
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor, n: int = 3):
        self.n = n
        super().__init__(low, high)

    def _func(self, x):
        return x ** self.n

    def _dfunc(self, x):
        return self.n * x ** (self.n - 1)

    def _ifunc(self, x):
        return x ** (1 / self.n)


class SinAngle(SuperUniform):
    r"""Sine angle distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.angular.SinAngle
    """

    @staticmethod
    def _func(x):
        return 1 - torch.cos(x)

    _dfunc = torch.sin

    @staticmethod
    def _ifunc(x):
        return torch.acos(1 - x)


class CosAngle(SuperUniform):
    r"""Cosine angle distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.angular.CosAngle
    """

    _func = torch.sin
    _dfunc = torch.cos
    _ifunc = torch.asin


class Joint(Distribution):
    r"""Joint distribution of marginals"""

    arg_constraints: dict = {}

    def __init__(self, marginals: List[Distribution]):
        super().__init__()

        self.marginals = marginals

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([sum(
            dist.event_shape.numel()
            for dist in self.marginals
        )])

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def sample(self, shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.sample(shape)
            y = y.view(shape + (-1,))
            x.append(y)

        return torch.cat(x, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        i, lp = 0, 0

        for dist in self.marginals:
            j = i + dist.event_shape.numel()
            y = x[..., i:j].view(shape + dist.event_shape)
            lp = lp + dist.log_prob(y)
            i = j

        return lp
