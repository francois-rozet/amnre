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
    ):
        super().__init__()

        # Prior
        bounds = torch.tensor([
            [10., 80.],  # mass1 [solar masses]
            [.125, 1.],  # mass ratio [/]
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

        # Simulator
        from .lfigw import waveform_generator as wfg
        from .lfigw.reduced_basis import SVDBasis

        path = os.path.dirname(inspect.getfile(wfg))
        path = os.path.join(path, 'events/GW150914/')

        wfd = wfg.WaveformDataset(spins_aligned=False, domain='RB')
        wfd.Nrb = n_rb
        wfd.approximant = 'IMRPhenomPv2'
        wfd.prior['distance'] = [100.0, 1000.0]
        wfd.prior['a_1'][1] = 0.88
        wfd.prior['a_2'][1] = 0.88

        ## Event
        wfd.load_event(path)

        with h5py.File(os.path.join(path, 'strain_FD_whitened.hdf5')) as f:
            strain = np.stack([f['H1'][:], f['L1'][:]])
            self.x_star = strain.astype(np.complex64)

        ## Basis
        try:
            wfd.basis = SVDBasis()
            wfd.basis.load(path)

            assert wfd.basis.n >= n_rb
            wfd.basis.truncate(n_rb)
        except:
            wfd.generate_reduced_basis(n_ref, 2 ** 10)
            wfd.basis.save(path)

        self.wfd = wfd

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        if mask is ...:
            mask = [True] * len(self.low)

        marginals = []

        for i, b in enumerate(mask):
            if not b:
                continue

            if i in [7, 8, 11]:  # [tilt_1, tilt_2, theta_jn]
                m = SinAngle(self.low[i], self.high[i])
            elif i == 14:  # declination
                m = CosAngle(self.low[i], self.high[i])
            else:
                m = Uniform(self.low[i], self.high[i])

            marginals.append(m)

        return Independent(Joint(marginals), 1)

    @cached_property
    def labels(self) -> List[str]:
        labels = [
            'm_1', 'q', r'\phi_c', 't_c', 'd_L',
            'a_1', 'a_2', r'\theta_1', r'\theta_2', r'\phi_{12}', r'\phi_{JL}',
            r'\theta_{JN}', r'\psi', r'\alpha', r'\delta',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def sample(self, theta: torch.Tensor, shape: torch.Size = ()) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        parameters = theta.view(-1, theta.shape[-1]).numpy().astype(np.float64)
        parameters[..., 1] *= parameters[..., 0]  # m_2 = m_1 * q

        x = np.stack(list(map(self._simulate, parameters)))
        x = self.postprocess(x)
        x = x.reshape(theta.shape[:-1] + x.shape[1:]).view(np.float32)

        if shape:
            x = np.broadcast_to(x, shape + x.shape)

        return torch.from_numpy(x)

    def _simulate(self, theta: np.ndarray) -> np.ndarray:
        x = self.wfd._generate_whitened_waveform(theta)
        x = np.stack(list(x.values()))

        return x.astype(np.complex64)

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        return self.wfd.basis.fseries_to_basis_coefficients(x) / self.wfd._noise_std

    def noise(self, shape: torch.Size = ()) -> torch.Tensor:
        size = shape + (len(self.wfd.detectors), self.wfd.Nrb * 2)
        noise = np.random.normal(size=size).astype(np.float32)

        return torch.from_numpy(noise)

    def events(self) -> Tuple[None, np.ndarray]:
        r""" x* """

        x = self.postprocess(self.x_star)
        x = x.view(np.float32)

        return None, x[None]


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
        return torch.Size()

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([sum(
            dist.batch_shape.numel()
            for dist in self.marginals
        )])

    def sample(self, shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.sample(shape)
            y = y.view(shape + (-1,))
            x.append(y)

        return torch.cat(x, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        i, lp = 0, []

        for dist in self.marginals:
            j = i + dist.batch_shape.numel()
            y = x[..., i:j].view(shape + dist.batch_shape)
            i = j

            lp.append(dist.log_prob(y).view(shape + (-1,)))

        return torch.cat(lp, dim=-1)
