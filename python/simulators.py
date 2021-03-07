#!/usr/bin/env python

import torch
import torch.nn as nn

from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Uniform,
)

from typing import Tuple, List


class Simulator(nn.Module):
    r"""Abstract Simulator"""

    def __init__(self):
        super().__init__()

    @property
    def prior(self) -> Distribution:
        r""" p(theta) """

        raise NotImplementedError()

    def likelihood(self, theta: torch.Tensor) -> Distribution:
        r""" p(x | theta) """

        raise NotImplementedError()

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r""" log p(theta) p(x | theta) """

        return self.prior.log_prob(theta) + self.likelihood(theta).log_prob(x)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample()

    def sample(self, batch_size: torch.Size = ()) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        theta = self.prior.sample(batch_size)
        x = self.__call__(theta)

        return theta, x


class SLCP(Simulator):
    r"""Simple Likelihood Complex Posterior"""

    def __init__(self, lim: float = 3.):
        super().__init__()

        self.register_buffer('low', -lim * torch.ones(5))
        self.register_buffer('high', lim * torch.ones(5))

    @property
    def prior(self) -> Distribution:
        r""" p(theta) """

        return Independent(Uniform(self.low, self.high), 1)

    def subprior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return Independent(Uniform(self.low[mask], self.high[mask]), 1)

    def likelihood(self, theta: torch.Tensor, eps: float = 1e-8) -> Distribution:
        r""" p(x | theta) """

        # Mean
        mu = theta[..., :2]

        # Covariance
        s1 = theta[..., 2] ** 2 + eps
        s2 = theta[..., 3] ** 2 + eps
        rho = theta[..., 4].tanh()

        cov = stack2d([
            [      s1 ** 2, rho * s1 * s2],
            [rho * s1 * s2,       s2 ** 2],
        ])

        # Repeat
        mu = mu.unsqueeze(-2).repeat_interleave(4, -2)
        cov = cov.unsqueeze(-3).repeat_interleave(4, -3)

        # Normal
        normal = MultivariateNormal(mu, cov)

        return Independent(normal, 1)


class MLCP(SLCP):
    r"""Mixture Likelihood Complex Posterior"""

    def __init__(self, lim: float = 3.):
        super().__init__()

        self.register_buffer('low', -lim * torch.ones(8))
        self.register_buffer('high', lim * torch.ones(8))

    def likelihood(self, theta: torch.Tensor, eps: float = 1e-8) -> Distribution:
        r""" p(x | theta) """

        # Rotation matrix
        alpha = theta[..., 0].sigmoid().asin()
        beta = theta[..., 1].sigmoid().acos()
        gamma = theta[..., 2].tanh().atan()

        zero = torch.zeros_like(alpha)
        one = torch.ones_like(alpha)

        Rz = stack2d([
            [alpha.cos(), -alpha.sin(), zero],
            [alpha.sin(),  alpha.cos(), zero],
            [       zero,         zero,  one],
        ])

        Ry = stack2d([
            [ beta.cos(), zero, beta.sin()],
            [       zero,  one,       zero],
            [-beta.sin(), zero, beta.cos()],
        ])

        Rx = stack2d([
            [ one,        zero,         zero],
            [zero, gamma.cos(), -gamma.sin()],
            [zero, gamma.sin(),  gamma.cos()],
        ])

        R = (Rz @ Ry @ Rx)[..., :2, :2]

        # Mean
        d = theta[..., 3] ** 2 + 1.
        mu = d[..., None, None] * R

        # Covariance
        s1 = theta[..., 4] ** 2 + eps
        s2 = theta[..., 5] ** 2 + eps
        rho = theta[..., 6].tanh()

        cov1 = stack2d([
            [      s1 ** 2, rho * s1 * s2],
            [rho * s1 * s2,       s2 ** 2],
        ])

        cov2 = stack2d([
            [1. / (s1 + 1.),           zero],
            [          zero, 1. / (s2 + 1.)],
        ])

        cov = torch.stack([cov1, cov2], dim=-3)

        # Mixture
        p = theta[..., 7].sigmoid()
        mix = torch.stack([p, 1. - p], dim=-1)

        # Repeat
        mix = mix.unsqueeze(-2).repeat_interleave(8, -2)
        mu = mu.unsqueeze(-3).repeat_interleave(8, -3)
        cov = cov.unsqueeze(-4).repeat_interleave(8, -4)

        # Normal
        normal = MixtureSameFamily(
            Categorical(mix),
            MultivariateNormal(mu, cov),
        )

        return Independent(normal, 1)


def stack2d(matrix: List[List[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([
        torch.stack(row, dim=-1)
        for row in matrix
    ], dim=-2)
