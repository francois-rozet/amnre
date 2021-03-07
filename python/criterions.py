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


class RELoss(nn.Module):
    r"""Ratio Estimator Loss (RELoss)"""

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(
        self,
        ratios: torch.Tensor,  # (N, *)
        mask: torch.BoolTensor,  # (T,)
    ) -> torch.Tensor:
        if torch.all(mask):
            l = self.bce(ratios, torch.ones_like(ratios))
        elif torch.all(~mask):
            l = self.bce(ratios, torch.zeros_like(ratios))
        else:
            l = torch.tensor(0., requires_grad=True)

        return l / len(ratios)


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


class RDLoss(nn.Module):
    r"""Ratio Distillation Loss (RDLoss)"""

    def __init__(self):
        super().__init__()

        self.se = nn.MSELoss(reduction='sum')

    def forward(
        self,
        ratios: torch.Tensor,  # (N, M)
        target: torch.Tensor,  # (N,)
    ) -> torch.Tensor:

        l = self.se(
            ratios.exp(),
            target.exp().unsqueeze(-1).expand(ratios.shape),
        )

        return l / len(target)
