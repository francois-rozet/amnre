#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class WeightedLoss(nn.Module):
    r"""Weighted Loss"""

    def __new__(cls, loss: nn.Module, weight: float = None):
        if weight is None:
            return loss
        else:
            return super().__new__(cls)

    def __init__(self, loss: nn.Module, weight: float):
        super().__init__()

        self.loss = loss
        self.weight = weight

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.weight * self.loss(*args, **kwargs)


class CompositeLoss(nn.Module):
    r"""Composite Loss"""

    def __init__(self, losses: List[nn.Module], weights: List[float] = None):
        super().__init__()

        if weights is None:
            weights = [None] * len(losses)

        self.losses = nn.ModuleList([
            WeightedLoss(l, w)
            for (l, w) in zip(losses, weights)
        ])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        l = torch.tensor(0.)

        for loss in self.losses:
            l = l + loss(*args, **kwargs)

        return l


class RMSELoss(nn.Module):
    r"""Root Mean Squared Error Loss (RMSELoss)"""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()

        self.epsilon = epsilon

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(input, target) + self.epsilon)


class RELoss(nn.Module):
    r"""Ratio Estimator Loss (RELoss)"""

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(
        self,
        ratio: torch.Tensor,  # r(theta | x)
        ratio_prime: torch.Tensor,  # r(theta' | x)
    ) -> torch.Tensor:
        l1 = self.bce(ratio, torch.ones_like(ratio))
        l0 = self.bce(ratio_prime, torch.zeros_like(ratio))

        return (l1 + l0) / ratio.size(0)


class RDLoss(nn.Module):
    r"""Ratio Distillation Loss (RDLoss)"""

    def forward(
        self,
        ratio: torch.Tensor,  # r(theta_a | x)
        target: torch.Tensor,  # r(theta | x)
    ) -> torch.Tensor:

        return F.mse_loss(ratio, target)


class SDLoss(nn.Module):
    r"""Score Distillation Loss (SDLoss)"""

    def forward(
        self,
        score: torch.Tensor,  # grad log r(theta_a | x)
        target: torch.Tensor,  # grad log r(theta | x)
    ) -> torch.Tensor:
        r"""Score Distillation Loss (SDLoss)"""

        return F.mse_loss(score, target, reduction='sum') / score.size(0)
