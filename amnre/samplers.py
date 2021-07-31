#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from itertools import islice
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
)

from typing import List, Union


class Transition:
    r"""Abstract Transition"""

    symmetric = False

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

    symmetric = True

    def __init__(self, sigma: torch.Tensor = 1.):
        super().__init__()

        self.sigma = sigma

    def distribution(self, x: torch.Tensor) -> Distribution:
        return Independent(Normal(
            x, torch.ones_like(x) * self.sigma
        ), 1)


class Sampler(data.IterableDataset):
    r"""Abstract sampler"""

    def __init__(self):
        super().__init__()

    def reference(self) -> torch.Tensor:
        r""" x_0 """

        raise NotImplementedError()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        r""" log p(x) """

        raise NotImplementedError()

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        r""" p(x) """

        return self.log_prob(x).exp()

    def __iter__(self): # -> torch.Tensor
        r""" x_i ~ p(x) """

        raise NotImplementedError()

    def __call__(
        self,
        steps: int,
        burn: int = 0,
        groupby: int = 1,
    ):  # -> torch.Tensor
        r""" (x_0, x_1, ..., x_n) ~ p(x) """

        seq = islice(self, burn, steps)

        if groupby > 1:
            buff = []

            for x in seq:
                buff.append(x)

                if len(buff) == groupby:
                    yield torch.cat(buff)
                    buff = []

            if buff:
                yield torch.cat(buff)
        else:
            yield from seq

    def histogram(
        self,
        bins: Union[int, List[int]],
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        r""" p(x) for all x """

        x = self.reference()

        # Shape
        B, D = x.shape

        if type(bins) is int:
            bins = [bins] * D

        # Create grid
        volume = 1.  # volume of one cell
        domains = []

        for l, h, b in zip(low, high, bins):
            step = (h - l) / b
            volume = volume * step

            dom = torch.linspace(l, h - step, b).to(step) + step / 2.
            domains.append(dom)

        grid = torch.stack(torch.meshgrid(*domains), dim=-1)
        grid = grid.view(-1, D).to(x)

        # Evaluate p(x) on grid
        p = []

        for x in grid.split(B):
            b = len(x)
            if b < B:
                x = F.pad(x, (0, 0, 0, B - b))

            p.append(self.prob(x)[:b])

        p = torch.cat(p).view(bins)

        # Scale w.r.t. cell volume
        p = p * volume

        return p


class MetropolisHastings(Sampler):
    r"""Metropolis-Hastings Algorithm

    Wikipedia:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    """

    def __init__(self, sigma: torch.Tensor = 1.):
        super().__init__()

        self.transition = NormalTransition(sigma)  # q(y | x)

    def __iter__(self): # -> torch.Tensor
        r""" x_i ~ p(x) """

        x = self.reference()

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

            if not self.transition.symmetric:
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


class LESampler(MetropolisHastings):
    r"""Likelihood Estimator Sampler (LESampler)"""

    def __init__(
        self,
        estimator: nn.Module,
        prior: Distribution,
        x: torch.Tensor,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.prior = prior

        self.batch_shape = (batch_size,)
        self.x = x.expand(self.batch_shape + x.shape)

    def reference(self) -> torch.Tensor:
        r""" theta_0 """

        return self.prior.sample(self.batch_shape)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        r""" log p(theta, x) """

        return self.estimator(theta, self.x) + self.prior.log_prob(theta)


class RESampler(LESampler):
    r"""Ratio Estimator Sampler (RESampler)"""
    pass


class PESampler(Sampler):
    r"""Posterior Estimator Sampler (PESampler)"""

    def __init__(
        self,
        estimator: nn.Module,
        x: torch.Tensor,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.estimator = estimator

        self.x = x[None]
        self.batch_shape = (batch_size,)
        self.y = x.expand(self.batch_shape + x.shape)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        r""" log p(theta | x) """

        return self.estimator(theta, self.y)

    def reference(self) -> torch.Tensor:
        r""" theta_0 """

        return next(iter(self))

    def __iter__(self): # -> torch.Tensor
        r""" theta ~ p(theta | x) """

        while True:
            yield self.estimator.sample(self.x, self.batch_shape)[0]
