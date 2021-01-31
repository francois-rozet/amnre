#!/usr/bin/env python

import torch
import torch.nn as nn


class MNRELoss(nn.Module):
    r"""Marginal Neural Ratio Estimator Loss (MNRELoss)"""

    def __init__(self, masks: torch.BoolTensor):
        super().__init__()

        self.register_buffer('masks', masks.float())  # (M, T)

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(
        self,
        ratios: torch.Tensor,  # (N, M)
        mask: torch.BoolTensor,  # (T,)
    ) -> torch.Tensor:
        match = self.masks @ mask.float()

        positive = match == self.masks.sum(dim=-1)
        negative = match == 0.

        p = ratios[..., positive]
        l_1 = self.bce(p, torch.ones_like(p))

        p = ratios[..., negative]
        l_0 = self.bce(p, torch.zeros_like(p))

        return (l_1 + l_0) / len(ratios)
