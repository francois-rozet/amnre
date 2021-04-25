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
    r"""Ratio Distillation Loss (RDLoss)

    (r(theta_a | x) - r(theta | x))^2

    Note:
        theta_b (theta / theta_a) has to be sampled from the prior p(theta_b)
    """

    def forward(
        self,
        ratio: torch.Tensor,  # log r(theta_a | x)
        target: torch.Tensor,  # log r(theta | x)
    ) -> torch.Tensor:
        ratio, target = ratio.exp(), target.exp()

        return F.mse_loss(ratio, target.detach())


class SDLoss(nn.Module):
    r"""Score Distillation Loss (SDLoss)

    (grad log r(theta_a | x) - grad log r(theta | x))^2

    Note:
        theta_b (theta / theta_a) has to be sampled from the prior p(theta_b)
    """

    def forward(
        self,
        theta: torch.Tensor,  # theta_a
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
        )[0]

        return F.mse_loss(score, target.detach())
