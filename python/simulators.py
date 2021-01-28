#!/usr/bin/env python

import torch
import torch.nn as nn

from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal

from typing import Tuple


class Simulator(nn.Module):
    r"""Abstract Simulator"""

    def __init__(self):
        super().__init__()

    @property
    def theta_star(self) -> torch.Tensor:
        r""" theta* """

        return self._theta_star

    @property
    def theta_size(self) -> torch.Size:
        return self.theta_star.size()

    @property
    def prior(self) -> Distribution:
        r""" p(theta) """

        raise NotImplementedError()

    def likelihood(self, theta: torch.Tensor) -> Distribution:
        r""" p(x | theta) """

        raise NotImplementedError()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample()

    def sample(self, batch_size: torch.Size = ()) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        theta = self.prior.sample(batch_size)
        x = self.__call__(theta)

        return theta, x


class Tractable(Simulator):
    r"""Tractable Simulator"""

    def __init__(self):
        super().__init__()

        self.register_buffer('_theta_star', torch.tensor([0.7, -2.9, -1., -0.9, 0.6]))
        self.register_buffer('_bound', 3. * torch.ones(5))

    @property
    def prior(self) -> Distribution:
        r""" p(theta) """

        return Uniform(-self._bound, self._bound)

    def likelihood(self, theta: torch.Tensor) -> Distribution:
        r""" p(x | theta) """

        mean = theta[..., :2]

        s1 = theta[..., 2] ** 2
        s2 = theta[..., 3] ** 2
        rho = theta[..., 4].tanh()

        covariance = torch.stack([
            torch.stack([s1 ** 2, rho * s1 * s2], dim=-1),
            torch.stack([rho * s1 * s2, s2 ** 2], dim=-1),
        ], dim=-1)

        return MultivariateNormal(mean, covariance)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r""" (x_1, ..., x_4) ~ π_{i=1}^4 p(x_i | theta) """

        x_i = self.likelihood(theta).sample((4,))

        return torch.cat(torch.unbind(x_i), dim=-1)
