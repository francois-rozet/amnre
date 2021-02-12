#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.data as data

from itertools import islice
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
)

from typing import Iterable

from simulators import Simulator


class Transition:
    r"""Abstract Transition"""

    def __init__(self):
        self._symmetric = False

    @property
    def symmertic(self):
        return self._symmetric

    def distribution(self, x: torch.Tensor) -> Distribution:
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        d = self.distribution(x)

        if y is None:
            return d.sample()
        else:
            return d.log_prob(y)


class NormalTransition(Transition):
    r"""Normal Transition"""

    def __init__(self, sigma: float = 1.):
        super().__init__()

        self._symmetric = True
        self.sigma = sigma

    def distribution(self, x: torch.Tensor) -> Distribution:
        return Independent(Normal(
            x, torch.ones_like(x) * self.sigma
        ), 1)


class MetropolisHastings(data.IterableDataset):
    r"""Metropolis-Hastings Algorithm

    Wikipedia:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    """

    @property
    def first(self) -> torch.Tensor:
        r""" x_0 """

        raise NotImplementedError()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        r""" log p(x) """

        raise NotImplementedError()

    @property
    def transition(self) -> Transition:
        r""" q(y | x) """

        return self._transition

    @torch.no_grad()
    def __iter__(self) -> torch.Tensor:
        x = self.first

        # p(x)
        p_x = self.log_prob(x)

        while True:
            # y ~ q(y | x)
            y = self.transition(x)

            # p(y)
            p_y = self.log_prob(y)

            #     p(y)   q(x | y)
            # a = ---- * --------
            #     p(x)   q(y | x)
            a = p_y - p_x

            if not self.transition.symmertic:
                a = a + self.transition(y, x) - self.transition(x, y)

            a = a.exp()

            # u in [0; 1]
            u = torch.rand(a.shape).to(a)

            # if u < a, x <- y
            # else x <- x
            mask = u < a

            x = torch.where(mask.unsqueeze(-1), y, x)
            p_x = torch.where(mask, p_y, p_x)

            yield x

    def __call__(self, n: int) -> Iterable[torch.Tensor]:
        return islice(self, n)


class TractableSampler(MetropolisHastings):
    r"""Tractable Simulator Sampler"""

    def __init__(
        self,
        simulator: Simulator,
        x: torch.Tensor,
        sigma: float = 1.,
    ):
        super().__init__()

        self.simulator = simulator
        self.x = x
        self.batch_shape = self.x.shape[:1]

        self._transition = NormalTransition(sigma)

    @property
    def first(self) -> torch.Tensor:
        r""" theta_0 """

        return self.simulator.prior.sample(self.batch_shape)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        r""" log p(theta, x) """

        return self.simulator.log_prob(theta, self.x)


class NRESampler(MetropolisHastings):
    r"""Neural Ratio Estimator Sampler (NRESampler)"""

    def __init__(
        self,
        prior: Distribution,
        nre: nn.Module,
        x: torch.Tensor,
        sigma: float = 1.,
    ):
        super().__init__()

        self.prior = prior
        self.nre = nre
        self.x = x
        self.batch_shape = self.x.shape[:1]

        self._transition = NormalTransition(sigma)

    @property
    def first(self) -> torch.Tensor:
        r""" theta_0 """

        return self.prior.sample(self.batch_shape)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        r""" log p(theta | x) """

        return self.nre(theta, self.x) + self.prior.log_prob(theta)
