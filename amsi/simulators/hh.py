#!/usr/bin/env python

import torch

from functools import cached_property
from torch.distributions import (
    Distribution,
    Independent,
    Uniform,
)

from typing import Tuple, List

from . import Simulator


class HH(Simulator):
    r"""Hodgkin-Huxley"""

    def __init__(self):
        super().__init__()

        low = torch.tensor([.5, 1e-4, 1e-4, 1e-4, 50., 40., 1e-4, 35.])
        high = torch.tensor([80., 15., .6, .6, 3000., 90., .15, 100.])

        self.register_buffer('low', low)
        self.register_buffer('high', high)

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return Independent(Uniform(self.low[mask], self.high[mask]), 1)

    @cached_property
    def labels(self) -> List[str]:
        labels = [r'g_{Na}', 'g_K', 'g_l', 'g_M', r'T_{max}', '-V_T', r'\sigma', '-E_l']
        labels = [f'${l}$' for l in labels]

        return labels
