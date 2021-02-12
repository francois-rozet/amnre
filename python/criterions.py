#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import List


class WeightedLoss(nn.Module):
    r"""Weighted Loss"""

    def __init__(self, losses: List[nn.Module], weights: List[float] = None):
        super().__init__()

        self.losses = nn.ModuleList(losses)

        if weights is None:
            weights = [1.] * len(losses)

        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        l = torch.tensor(0.)

        for loss, w in zip(self.losses, self.weights):
            l = l + w * loss(*args, **kwargs)

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


class RDLoss(nn.Module):
    r"""Ratio Distillation Loss (RDLoss)"""

    def __init__(self, masks: torch.BoolTensor):
        super().__init__()

        unions = masks.unsqueeze(1) + masks
        intersections = masks.unsqueeze(1) * masks

        # if subsets[i, j]: mask[j] in mask[i]
        self.subsets = intersections.sum(-1) == masks.sum(-1)
        self.m = len(self.subsets)

        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        ratios: torch.Tensor,  # (N, M)
        mask: torch.BoolTensor,  # (T,)
    ) -> torch.Tensor:
        l = torch.tensor(0., requires_grad=True)

        if torch.all(~mask):
            ratios = ratios.exp()

            for i in range(self.m):
                for j in range(self.m):
                    if i != j and self.subsets[i, j]:
                        l = l + self.mse(ratios[:, i].detach(), ratios[:, j])

        return l
