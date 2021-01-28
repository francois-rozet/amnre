#!/usr/bin/env python

import torch
import torch.nn as nn


class ForkLoss(nn.Module):
    r"""Fork Loss (FL)"""

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,  # (N, H)
        true_masks: torch.BoolTensor,  # (N, T)
        head_masks: torch.BoolTensor,  # (H, T)
    ) -> torch.Tensor:
        matches = true_masks.float() @ head_masks.float().t()

        positive = matches == head_masks.sum(dim=-1)
        negative = matches == 0.

        pred = logits[positive]
        l_1 = self.bce(pred, torch.ones_like(pred))

        pred = logits[negative]
        l_0 = self.bce(pred, torch.zeros_like(pred))

        return l_1 + l_0
