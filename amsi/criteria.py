#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


def reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'sum':
        x = x.sum()
    elif reduction == 'mean':
        x = x.mean()
    elif reduction == 'batchmean':
        x = x.sum() / x.size(0)

    return x


class MSELoss(nn.Module):
    r"""Mean Squared Error (MSE) Loss"""

    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        e = F.mse_loss(input, target, reduction='none')

        return reduce(e, self.reduction)


class NLLWithLogitsLoss(nn.Module):
    r"""Negative Log-Likelihood (NLL) With Logits Loss

    - log(x)
    """

    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        ll = F.logsigmoid(input)  # log-likelihood

        if weight is not None:
            ll = weight * ll

        return -reduce(ll, self.reduction)


class FocalWithLogitsLoss(nn.Module):
    r"""Focal With Logits Loss

    - (1 - x)^gamma log(x)

    References:
        [1] Focal Loss for Dense Object Detection
        (Lin et al., 2017)
        https://arxiv.org/abs/1708.02002

        [2] Calibrating Deep Neural Networks using Focal Loss
        (Mukhoti et al., 2020)
        https://arxiv.org/abs/2002.09437
    """

    def __init__(self, gamma: float = 2., reduction: str = 'batchmean'):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        ll = F.logsigmoid(input)  # log-likelihood
        fc = -(1 - ll.exp()) ** self.gamma * ll  # focal

        if weight is not None:
            fc = weight * fc

        return reduce(fc, self.reduction)


class PeripheralWithLogitsLoss(FocalWithLogitsLoss):
    r"""Peripheral With Logits Loss

    - (1 - x^gamma) log(x)

    Note:
        This is an adaptation of the Focal Loss.
    """

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        ll = F.logsigmoid(input)  # log-likelihood
        pp = -(1 - (ll * self.gamma).exp()) * ll  # peripheral

        if weight is not None:
            pp = weight * pp

        return reduce(pp, self.reduction)


class QSWithLogitsLoss(nn.Module):
    r"""Quadratic Score (QS) With Logits Loss

    (1 - x)^2

    References:
        https://en.wikipedia.org/wiki/Scoring_rule
    """

    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()

        self.reduction = reduction

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        qs = F.sigmoid(-input) ** 2

        if weight is not None:
            qs = weight * qs

        return reduce(qs, self.reduction)


class RRLoss(MSELoss):
    r"""Ratio Regression (RR) Loss

    (r - r*)^2 or (1 / r - 1 / r*)^2
    """

    def forward(
        self,
        ratio: torch.Tensor,  # +- log r
        target: torch.Tensor,  # +- log r*
    ) -> torch.Tensor:
        ratio, target = ratio.exp(), target.detach().exp()

        while target.dim() < ratio.dim():
            target = target[..., None]

        return super().forward(ratio, target.expand(ratio.shape))


class SRLoss(MSELoss):
    r"""Score Regression (SR) Loss

    (grad log r(theta_a | x) - grad log r(theta | x))^2
    """

    def forward(
        self,
        theta: torch.Tensor,  # theta
        ratio: torch.Tensor,  # log r(theta_a | x)
        target: torch.Tensor,  # log r(theta | x)
    ) -> torch.Tensor:
        score = torch.autograd.grad(  # grad log r(theta_a | x)
            ratio, theta,
            torch.ones_like(ratio),
            create_graph=True,
        )[0]

        target = torch.autograd.grad(  # grad log r(theta | x)
            target, theta,
            torch.ones_like(target),
        )[0].detach()

        while target.dim() < score.dim():
            target = target[..., None]

        return super().forward(score, target.expand(score.shape))
