#!/usr/bin/env python

import math
import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
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
    r"""LIGO's Noise Spectral Density (NSD) and its standard deviation"""

    from pycbc.psd import aLIGOZeroDetHighPower

    psd = aLIGOZeroDetHighPower(length, delta_f, cutoff_freq)

    idx = int(psd.duration * cutoff_freq)
    sigma = np.sqrt(psd.data / psd.delta_f) / 2
    sigma[:idx] = sigma[idx]
    sigma[-1] = sigma[-2]

    return psd, sigma


def lal_spins(
    mass1: float,
    mass2: float,
    coa_phase: float,
    a_1: float,
    a_2: float,
    tilt_1: float,
    tilt_2: float,
    phi_12: float,
    phi_jl: float,
    theta_jn: float,
    ref_freq: float = 20.,  # Hz
) -> dict:
    """Convert to parameters to spins

    References:
        https://github.com/lscsoft/bilby/blob/master/bilby/gw/conversion.py#L53
    """

    from lal import MSUN_SI
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions as transform

    if (a_1 == 0.0 or tilt_1 in [0, np.pi]) and (a_2 == 0.0 or tilt_2 in [0, np.pi]):
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            theta_jn,
            0., 0., a_1 * np.cos(tilt_1),
            0., 0., a_2 * np.cos(tilt_2),
        )
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = transform(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
            mass1 * MSUN_SI, mass2 * MSUN_SI, ref_freq, coa_phase,
        )

    return {
        'inclination': iota,
        'spin_1x': spin_1x,
        'spin_2x': spin_2x,
        'spin_1y': spin_1y,
        'spin_2y': spin_2y,
        'spin_1z': spin_1z,
        'spin_2z': spin_2z,
    }


def generate_waveform(
    theta: np.ndarray,
    approximant: str = 'IMRPhenomPv2',
    duration: float = 4.,  # s
    sample_rate: int = 2048,  # Hz
    ref_freq: float = 20.,  # Hz
    detectors: List[str] = ['H1', 'L1'],
) -> np.ndarray:
    r"""Waveform generation in the frequency domain

    References:
    	https://github.com/gwastro/pycbc
        https://github.com/stephengreen/lfi-gw
    """

    try:
        from pycbc.waveform import get_td_waveform
        from pycbc.detector import Detector
    except:
        shape = len(detectors), int(duration * sample_rate / 2) + 1
        return np.zeros(shape, dtype=complex)

    theta = theta.astype(float)

    # Waveform simulation
    wav_args = {
        ## Static
        'approximant': approximant,
        'delta_t': 1 / sample_rate,
        'f_lower': ref_freq,
        'f_ref': ref_freq,
        ## Variables
        'mass1': theta[0],
        'mass2': theta[1],
        'coa_phase': theta[2],
        'distance': theta[3],
    }

    spins = lal_spins(*theta[0:3], *theta[4:11], ref_freq)

    hp, hc = get_td_waveform(**wav_args, **spins)

    # Projection on detectors
    proj_args = {
        ## Static
        'method': 'constant',
        'reference_time': 921726855,  # 18/03/1999 - 03:14:15
        ## Variables
        'polarization': theta[11],
        'ra': theta[12],
        'dec': theta[13],
    }

    signals = {
        det: Detector(det).project_wave(hp, hc, **proj_args)
        for det in detectors
    }

    # Fast Fourier Transform
    length = int(duration * sample_rate)

    for det, s in signals.items():
        if len(s) < length:
            s.prepend_zeros(length - len(s))
        else:
            s = s[len(s) - length:]

        signals[det] = s.to_frequencyseries()

    # Export
    x = np.stack([s.data for s in signals.values()])

    return x


def whiten_waveform(
    x: np.ndarray,
    duration: float = 4.,  # s
    cutoff_freq: float = 20.,  # Hz
) -> np.ndarray:
    """Whitening in the frequency domain

    References:
        http://pycbc.org/pycbc/latest/html/pycbc.types.html#pycbc.types.timeseries.TimeSeries.whiten
    """

    _, sigma = ligo_nsd(x.shape[-1], 1 / duration, cutoff_freq)

    return x / sigma


def generate_noise(
    shape: Tuple[int, ...] = (),
    duration: float = 4.,  # s
    sample_rate: int = 2048,  # Hz
    cutoff_freq: float = 20.,  # Hz
    detectors: list = ['H1', 'L1'],
) -> np.ndarray:
    r"""Detector noise generation in the frequency domain

    References:
        https://pycbc.org/pycbc/latest/html/pycbc.noise.html#pycbc.noise.gaussian.frequency_noise_from_psd
    """

    shape = shape + (len(detectors), int(duration * sample_rate / 2) + 1)

    psd, _ = ligo_nsd(shape[-1], 1 / duration, cutoff_freq)
    mask = psd != 0

    real = np.random.normal(size=shape)
    imag = np.random.normal(size=shape)

    noise = (real + 1j * imag) * mask

    return noise


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
        https://github.com/stephengreen/lfi-gw
    """

    def __init__(self, fiducial: bool = False, basis: np.ndarray = None):
        super().__init__()

        self.fiducial = fiducial
        self.basis = basis

        bounds = torch.tensor([
            [10., 80.],  # mass1 [solar masses]
            [10., 80.],  # mass2 [solar masses]
            [0., 2 * math.pi],  # coalesence phase [rad]
            [100., 1000.],  # luminosity distance [megaparsec]
            [0., 0.99],  # a_1 [/]
            [0., 0.99],  # a_2 [/]
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

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        if mask is ...:
            mask = [True] * len(self.low)

        marginals = []

        for i, b in enumerate(mask):
            if not b:
                continue

            if i == 3:  # luminosity distance
                m = PowerLaw(self.low[i], self.high[i], n=3)
            elif i in [6, 7, 10]:  # [tilt_1, tilt_2, theta_jn]
                m = SinAngle(self.low[i], self.high[i])
            elif i == 13:  # declination
                m = CosAngle(self.low[i], self.high[i])
            else:
                m = Uniform(self.low[i], self.high[i])

            marginals.append(m)

        return Independent(Joint(marginals), 1)

    @property
    def labels(self) -> List[str]:
        labels = [
            'm_1', 'm_2', r'\phi_c', 'd_L',
            'a_1', 'a_2', r'\theta_1', r'\theta_2', r'\phi_{12}', r'\phi_{JL}',
            r'\theta_{JN}', r'\psi', r'\alpha', r'\delta',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        if self.fiducial:
            theta[..., 3] = self.high[3]

        seq = theta.view(-1, theta.size(-1)).numpy()

        with Pool(cpu_count()) as p:
            x = p.map(generate_waveform, iter(seq))

        x = np.stack(x)
        x = whiten_waveform(x)

        if self.basis is not None:
            x = self.reduction(x)

        x = x.reshape(theta.shape[:-1] + x.shape[1:])

        return torch.from_numpy(x)

    def reduction(self, x: np.ndarray) -> np.ndarray:
        x = x @ self.basis
        x = x.view(x.real.dtype).reshape(x.shape + (2,))
        x = x.astype(np.float32)

        return x

    def noise(self, shape: torch.Size = ()) -> torch.Tensor:
        noise = generate_noise(shape)

        if self.basis is not None:
            noise = self.reduction(noise)

        return torch.from_numpy(noise)


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
