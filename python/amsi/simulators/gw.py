#!/usr/bin/env python

import h5py
import math
import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache, partial
from multiprocessing import cpu_count, Pool
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import List, Tuple

from . import Simulator


@lru_cache(None)
def ligo_nsd(length: int, delta_f: float, cutoff_freq: float) -> tuple:
    r"""LIGO's Noise Spectral Density (NSD) and its standard deviation

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.noise.html#pycbc.noise.gaussian.frequency_noise_from_psd
    """

    from pycbc.psd import aLIGOZeroDetHighPower

    psd = aLIGOZeroDetHighPower(length, delta_f, cutoff_freq)
    psd.data[-1] = psd.data[-2]

    idx = int(psd.duration * cutoff_freq)
    sigma = np.sqrt(psd / psd.delta_f) / 2
    sigma[:idx] = sigma[idx]

    return psd, sigma


def generate_waveform(
    theta: np.ndarray,
    approximant: str = 'IMRPhenomPv2',
    duration: float = 4.,  # s
    sample_rate: int = 2048,  # Hz
    cutoff_freq: float = 20.,  # Hz
    detectors: list = ['H1', 'L1'],
    noisy: bool = True,
) -> np.ndarray:
    r"""Waveform generation in the frequency domain

    References:
        https://github.com/stephengreen/lfi-gw
    """

    try:
        from pycbc.waveform import get_td_waveform
        from pycbc.detector import Detector
        from pycbc.noise import frequency_noise_from_psd
    except:
        shape = len(detectors), int(duration * sample_rate / 2) + 1
        return np.zeros(shape)

    theta = theta.astype(float)

    # Waveform simulation
    wav_args = {
        ## Static
        'approximant': approximant,
        'delta_t': 1 / sample_rate,
        'f_lower': cutoff_freq,
        ## Variables
        'distance': theta[0],
        'mass1': theta[1],
        'mass2': theta[2],
        'spin1z': theta[3],
        'spin2z': theta[4],
        'coa_phase': theta[5],
        'inclination': theta[6],
    }

    hp, hc = get_td_waveform(**wav_args)

    # Projection on detectors
    proj_args = {
        ## Static
        'method': 'constant',
        'reference_time': 921726855,  # 18/03/1999 - 03:14:15
        ## Variables
        'dec': theta[7],
        'ra': theta[8],
        'polarization': theta[9],
    }

    signals = {}
    for det in detectors:
        signals[det] = Detector(det).project_wave(hp, hc, **proj_args)

    # Fast Fourier Transform
    length = int(duration * sample_rate)

    for det, s in signals.items():
        if len(s) < length:
            s.prepend_zeros(length - len(s))
        else:
            s = s[len(s) - length:]

        signals[det] = s.to_frequencyseries()

    # Noise
    s = signals[det]
    psd, sigma = ligo_nsd(len(s), s.delta_f, cutoff_freq)

    for det, s in signals.items():
        if noisy:
            s += frequency_noise_from_psd(psd)

        s /= sigma  # whitening

    # Export
    x = np.stack([s.data for s in signals.values()])
    x = x.astype(np.complex64)

    return x


def svd_basis(x: np.ndarray, n: int) -> np.ndarray:
    r"""Singular Value Decomposition (SVD) basis reduction

    References:
        https://github.com/stephengreen/lfi-gw/blob/master/lfigw/reduced_basis.py
    """

    from sklearn.utils.extmath import randomized_svd

    _, _, Vt = randomized_svd(x, n)
    Vt = Vt.astype(x.dtype)
    return Vt.T.conj()


class GW(Simulator):
    r"""Gravitational Waves

    References:
        https://github.com/gwastro/pycbc
    """

    def __init__(self, fiducial: bool = False, basis: np.ndarray = None):
        super().__init__()

        self.fiducial = fiducial
        self.basis = basis

        bounds = torch.tensor([
            # 3-rd power law
            [10., 1000.],  # distance [megaparsec]
            # uniform
            [10., 80.],  # mass1 [solar masses]
            [10., 80.],  # mass2 [solar masses]
            [-.9, .9],  # spin1z [/]
            [-.9, .9],  # spin2z [/]
            [0., 2 * math.pi],  # coa_phase [rad]
            # sin angle
            [0., math.pi],  # inclination [rad]
            # cos angle
            [-math.pi / 2, math.pi / 2],  # dec [rad]
            # uniform
            [0., math.pi],  # ra [rad]
            [0. , 2 * math.pi],  # polarization [rad]
        ])

        self.register_buffer('low', bounds[:, 0])
        self.register_buffer('high', bounds[:, 1])

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        if mask is ...:
            mask = torch.tensor(True).expand(self.low.shape)

        marginals = []

        if mask[0]:
            marginals.append(PowerLaw(self.low[0], self.high[0], n=3))

        if torch.any(mask[1:6]):
            marginals.append(Uniform(self.low[1:6][mask[1:6]], self.high[1:6][mask[1:6]]))

        if mask[6]:
            marginals.append(SinAngle(self.low[6], self.high[6]))

        if mask[7]:
            marginals.append(CosAngle(self.low[7], self.high[7]))

        if torch.any(mask[8:]):
            marginals.append(Uniform(self.low[8:][mask[8:]], self.high[8:][mask[8:]]))

        return Independent(Joint(marginals), 1)

    @property
    def labels(self) -> List[str]:
        labels = ['d_L', 'm_1', 'm_2', 's_1', 's_2', '\\Phi', '\\theta', '\\delta', '\\alpha', '\\Psi']
        labels = [f'${l}$' for l in labels]

        return labels

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        if self.fiducial:
            fun = partial(generate_waveform, noisy=False)
            theta[:, 0] = self.high[0]
        else:
            fun = generate_waveform

        seq = theta.view(-1, theta.size(-1)).cpu().numpy()

        with Pool(cpu_count()) as p:
            x = p.map(fun, iter(seq))

        x = np.stack(x)
        x = x.reshape(theta.shape[:-1] + x.shape[1:])

        if self.basis is not None:
            x = x @ self.basis
            x = x.view(x.real.dtype).reshape(x.shape + (2,))

        return torch.from_numpy(x).to(theta.device)


class BasisGW(Simulator):
    def __new__(self, n: int, m: int):
        sim = GW(fiducial=True)

        _, x = sim.sample((m,))
        x = x.view(-1, x.size(-1)).numpy()

        sim.fiducial = False
        sim.basis = svd_basis(x, n)

        return sim


class PowerLaw(Uniform):
    r"""Power law distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.power_law.UniformPowerLaw
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor, n: int = 3):
        super().__init__(low ** n, high ** n)

        self.n = n
        self.log_n = math.log(n)
        self.log_volume = (self.high - self.low).log()

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        return super().sample(sample_shape) ** (1 / self.n)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return value.log() * (self.n - 1) + self.log_n - self.log_volume


class SinAngle(Uniform):
    r"""Sine angle distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.angular.SinAngle
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__(high.cos(), low.cos())

        self.log_volume = (self.high - self.low).log()

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        return super().sample(sample_shape).acos()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return value.sin().log() - self.log_volume


class CosAngle(Uniform):
    r"""Cosine angle distribution

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.distributions.html#pycbc.distributions.angular.CosAngle
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__(low.sin(), high.sin())

        self.log_volume = (self.high - self.low).log()

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        return super().sample(sample_shape).asin()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return value.cos().log() - self.log_volume


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

    def sample(self, sample_shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.sample(sample_shape)
            y = y.view(sample_shape + (-1,))
            x.append(y)

        return torch.cat(x, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        sample_shape = x.shape[:-1]
        i, lp = 0, []

        for dist in self.marginals:
            j = i + dist.batch_shape.numel()
            y = x[..., i:j].view(sample_shape + dist.batch_shape)
            i = j

            lp.append(dist.log_prob(y).view(sample_shape + (-1,)))

        return torch.cat(lp, dim=-1)
