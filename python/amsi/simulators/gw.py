#!/usr/bin/env python

import h5py
import math
import numpy as np
import torch
import torch.nn as nn

from multiprocessing import cpu_count, Pool
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import Tuple, List

from . import Simulator


class GW(Simulator):
    r"""Gravitational Waves

    References:
        https://github.com/gwastro/pycbc
    """

    def __init__(self):
        super().__init__()

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

    @staticmethod
    def generate(theta: np.ndarray) -> np.ndarray:
        from pycbc.waveform import get_td_waveform
        from pycbc.detector import Detector
        from pycbc.psd import aLIGOZeroDetHighPower
        from pycbc.noise import noise_from_psd

        theta = theta.astype(float)

        # Waveform simulation
        sample_rate = 2048

        wav_args = {
            ## Static
            'approximant': 'IMRPhenomPv2',
            'delta_t': 1 / sample_rate,  # s
            'f_lower': 20,  # Hz
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
        det_args = {
            ## Static
            'reference_time': 921726855,  # 18/03/1999 - 03:14:15
            ## Variables
            'dec': theta[7],
            'ra': theta[8],
            'polarization': theta[9],
        }

        td_length = 128 * sample_rate  # 128s

        signals = {}
        for det in ['H1', 'L1']:
            signals[det] = Detector(det).project_wave(hp, hc, **det_args)
            signals[det].resize(td_length)

        # PSD
        psd_args = {
            ## Static
            'length': td_length * 2 + 1,  # ?
            'delta_f': 1 / 128,  # Hz
            'low_freq_cutoff': wav_args['f_lower'],
        }

        psd = aLIGOZeroDetHighPower(**psd_args)

        # Strain
        noise_args = {
            ## Static
            'length': td_length,
            'delta_t': wav_args['delta_t'],
            'psd': psd,
        }

        for det, s in signals.items():
            noise = noise_from_psd(**noise_args)
            noise.start_time = s.start_time
            signals[det] = noise.add_into(s)

        # Whitening
        white_args = {
            ## Static
            'segment_duration': 4,  # s
            'max_filter_duration': 4,  # s
            'remove_corrupted': False,
        }

        for det, s in signals.items():
            signals[det] = s.whiten(**white_args)

        # High-pass filter
        high_args = {
            ## Static
            'frequency': wav_args['f_lower'],
            'remove_corrupted': False,
            'order': 512,
        }

        for det, s in signals.items():
            signals[det] = s.highpass_fir(**high_args)

        # Output
        length = 4 * sample_rate  # 4s
        x = np.stack([s.numpy()[:length] for s in signals.values()])
        x = x.astype(np.float32)

        return x

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        import pycbc

        theta_np = theta.view(-1, theta.size(-1)).cpu().numpy()
        with Pool(cpu_count()) as p:
            x = p.map(self.generate, iter(theta_np))

        x = torch.from_numpy(np.stack(x))
        x = x.view(theta.shape[:-1] + x.shape[1:]).to(theta)

        return x


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
