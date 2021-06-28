#!/usr/bin/env python

import math
import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache, cached_property
from multiprocessing import cpu_count, Pool
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import List, Tuple

from . import Simulator


@lru_cache(None)
def ligo_detector(name: str):
    r"""Get LIGO detector"""

    from pycbc.detector import Detector

    return Detector(name)


@lru_cache(None)
def event_time(name: str = 'GW150914') -> float:
    r"""Get event's GPS time"""

    from pycbc.catalog import Merger

    return Merger(name).data['GPS']


def tuckey_window(
    duration: int,  # s
    sample_rate: float,  # Hz
    roll_off: float = 0.4,  # /
) -> np.ndarray:
    r"""Tuckey window function

    References:
        https://en.wikipedia.org/wiki/Window_function
    """

    from scipy.signal import tukey

    length = int(duration * sample_rate)
    alpha = 2 * roll_off / duration

    return tukey(length, alpha)


def event_nsd(
    name: str = 'GW150914',
    detectors: Tuple[str] = ('H1', 'L1'),
    duration: float = 8.,  # s
    segment: float = 1024.,  # s
) -> np.ndarray:
    r"""Get event's Noise Spectral Density (NSD)"""

    from gwpy.timeseries import TimeSeries  # /!\ export GWPY_RCPARAMS=0
    from pycbc.psd import welch

    # Fetch
    time = event_time(name) - duration

    strains = {
        ifo: TimeSeries.fetch_open_data(ifo, time - segment, time, cache=True).to_pycbc()
        for ifo in detectors
    }

    # Welch
    for s in strains.values():
        w = tuckey_window(duration, s.sample_rate)
        factor = (w ** 2).mean()
        break

    psds = {
        ifo: welch(s, len(w), len(w), window=w, avg_method='median') * factor
        for (ifo, s) in strains.items()
    }

    return np.stack([p.data for p in psds.values()])


def event_dft(
    name: str = 'GW150914',
    detectors: Tuple[str] = ('H1', 'L1'),
    duration: float = 8.,  # s
    shift: float = 2.,  # s
) -> np.ndarray:
    r"""Get event's Discrete Fourier Transform (DFT)"""

    from gwpy.timeseries import TimeSeries

    # Fetch
    time = event_time(name) + shift

    strains = {
        ifo: TimeSeries.fetch_open_data(ifo, time - duration, time, cache=True).to_pycbc()
        for ifo in detectors
    }

    for s in strains.values():
        sample_rate = s.sample_rate
        break

    # Discrete Fourier Transform
    for s in strains.values():
        w = tuckey_window(duration, s.sample_rate)
        break

    dfts = np.stack([
        (s * w).to_frequencyseries().cyclic_time_shift(shift).data
        for s in strains.values()
    ])

    return dfts


def lal_spins(
    m_1: float,
    m_2: float,
    phi_c: float,
    a_1: float,
    a_2: float,
    tilt_1: float,
    tilt_2: float,
    phi_12: float,
    phi_jl: float,
    theta_jn: float,
    ref_freq: float = 20.,  # Hz
) -> dict:
    """Convert parameters to LAL spins

    References:
        https://github.com/lscsoft/bilby/blob/master/bilby/gw/conversion.py#L53
    """

    from lal import MSUN_SI
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions as transform

    if (a_1 == 0.0 or tilt_1 in [0, np.pi]) and (a_2 == 0.0 or tilt_2 in [0, np.pi]):
        iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = (
            theta_jn,
            0., 0., a_1 * np.cos(tilt_1),
            0., 0., a_2 * np.cos(tilt_2),
        )
    else:
        iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = transform(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
            m_1 * MSUN_SI, m_2 * MSUN_SI, ref_freq, phi_c,
        )

    return {
        'inclination': iota,
        'spin1x': spin1x,
        'spin2x': spin2x,
        'spin1y': spin1y,
        'spin2y': spin2y,
        'spin1z': spin1z,
        'spin2z': spin2z,
    }


def generate_waveform(
    theta: np.ndarray,
    approximant: str = 'IMRPhenomPv2',
    duration: float = 8.,  # s
    sample_rate: float = 2048.,  # Hz
    ref_freq: float = 20.,  # Hz
    event: str = 'GW150914',
    detectors: Tuple[str] = ('H1', 'L1'),
) -> np.ndarray:
    r"""Waveform generation in the frequency domain

    References:
        https://github.com/gwastro/pycbc
        https://github.com/stephengreen/lfi-gw
    """

    from pycbc.waveform import get_fd_waveform

    theta = theta.astype(float)

    # Waveform simulation
    m_1, q, phi_c, t_c, d_l = theta[:5]
    m_2 = m_1 * q

    wav_args = {
        ## Static
        'approximant': approximant,
        'delta_f': 1 / duration,
        'f_lower': ref_freq,
        'f_ref': ref_freq,
        'f_final': sample_rate / 2,
        ## Variables
        'mass1': m_1,
        'mass2': m_2,
        'coa_phase': phi_c,
        'distance': d_l,
    }

    spins = lal_spins(m_1, m_2, phi_c, *theta[5:12], ref_freq)
    wav_args.update(spins)

    hp, hc = get_fd_waveform(**wav_args)

    # Projection on detectors
    psi, alpha, delta = theta[12:]
    ref_time = event_time(event)

    signals = []

    for ifo in detectors:
        detector = ligo_detector(ifo)

        fp, fc = detector.antenna_pattern(alpha, delta, psi, ref_time)
        hd = fp * hp + fc * hc

        dt = detector.time_delay_from_earth_center(alpha, delta, ref_time)
        hd = hd.cyclic_time_shift(dt + t_c)

        signals.append(hd)

    return np.stack([s.data for s in signals])


def crop_dft(
    dft: np.ndarray,
    duration: float = 8.,  # s
    sample_rate: float = 2048.,  # Hz
    ref_freq: float = 20.,  # Hz
) -> np.ndarray:
    r"""Crop Discrete Fourier Transform (DFT)"""

    return dft[..., int(duration * ref_freq):int(duration * sample_rate / 2) + 1]


def whiten_dft(
    dft: np.ndarray,
    psd: np.ndarray,
    duration: float = 8.,  # s
) -> np.ndarray:
    r"""Whiten Discrete Fourier Transform (DFT) w.r.t. Power Spectral Density (PSD)"""

    return 2 * dft / np.sqrt(psd * duration)


def generate_noise(
    shape: Tuple[int, ...] = (),
    duration: float = 8.,  # s
    sample_rate: float = 2048.,  # Hz
    ref_freq: float = 20.,  # Hz
    detectors: Tuple[str] = ('H1', 'L1'),
) -> np.ndarray:
    r"""Generate unit gaussian noise in the frequency domain"""

    length = int(duration * sample_rate / 2) + 1 - int(duration * ref_freq)
    shape = shape + (len(detectors), length)

    real = np.random.normal(size=shape)
    imag = np.random.normal(size=shape)

    return real + 1j * imag


def svd_basis(x: np.ndarray, n: int = 256) -> np.ndarray:
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
            [.1, 1.],  # mass ratio [/]
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

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        if self.fiducial:
            theta[..., 3] = 0.
            theta[..., 4] = self.high[4]

        seq = theta.view(-1, theta.size(-1)).numpy()

        with Pool(cpu_count()) as p:
            x = p.map(generate_waveform, iter(seq))

        x = crop_dft(np.stack(x))
        x = whiten_dft(x, self.psd)

        if self.basis is not None:
            x = self.reduction(x)

        x = x.reshape(theta.shape[:-1] + x.shape[1:])

        return torch.from_numpy(x)

    @cached_property
    def psd(self) -> np.ndarray:
        return crop_dft(event_nsd())

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

    def events(self) -> Tuple[None, np.ndarray]:
        r""" x* """

        x = crop_dft(event_dft())
        x = whiten_dft(x, self.psd)

        if self.basis is not None:
            x = self.reduction(x)

        return None, x[None]


class SuperUniform(Uniform):
    r"""Abstract more-than-uniform distribution"""

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__(self._func(low), self._func(high))

        self.log_volume = (self.high - self.low).log()
        self.low_, self.high_ = low, high

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        return self._ifunc(super().sample(sample_shape))

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
