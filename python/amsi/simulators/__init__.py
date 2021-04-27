#!/usr/bin/env python

import torch
import torch.nn as nn

from functools import cached_property
from torch.distributions import Distribution
from typing import Tuple, List


Distribution.set_default_validate_args(False)


class Simulator(nn.Module):
    r"""Abstract Simulator"""

    @property
    def prior(self) -> Distribution:
        r""" p(theta) """

        return self.masked_prior(...)

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        raise NotImplementedError()

    @cached_property
    def labels(self) -> List[str]:  # parameters' labels
        theta_size = self.prior.sample().numel()
        labels = [f'$\\theta_{{{i}}}$' for i in range(1, theta_size + 1)]

        return labels

    def likelihood(self, theta: torch.Tensor) -> Distribution:
        r""" p(x | theta) """

        raise NotImplementedError()

    @cached_property
    def tractable(self) -> bool:
        theta = self.prior.sample()

        try:
            lkh = self.likelihood(theta)
            return True
        except NotImplementedError as e:
            return False

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r""" log p(theta) p(x | theta) """

        return self.prior.log_prob(theta) + self.likelihood(theta).log_prob(x)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample()

    def sample(self, sample_shape: torch.Size = ()) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        theta = self.prior.sample(sample_shape)
        x = self(theta)

        return theta, x
