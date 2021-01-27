#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import Callable, List


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        input_size: The input size.
        output_size: The output size.
        num_layers: The number of layers.
        hidden_size: The size of hidden layers.
        bias: Whether to use bias.
        dropout: The dropout rate.
        activation: A callable returning an activation layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        num_layers: int = 2,
        hidden_size: int = 64,
        bias: bool = True,
        dropout: float = 0.,
        activation: Callable[[int], nn.Module] = nn.PReLU,
    ):
        if dropout > 0.:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        layers = [nn.Linear(input_size, hidden_size, bias), activation(hidden_size), dropout]

        for i in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size, bias), activation(hidden_size), dropout])

        layers.append(nn.Linear(hidden_size, output_size, bias))

        super().__init__(*layers)

        self.input_size = input_size
        self.output_size = output_size


class ForkClassifier(nn.Module):
    r"""Fork Classifier (FC)

                    ---> f_a(y_a, z(x))
                   /
    x ---> z(x), y ----> f_b(y_b, z(x))
                   \
                    ---> f_c(y_c, z(x))

    Args:
        encoder: A forward module to process `x`.
        masks: The masks of each considered subset of `y`.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(self, encoder: nn.Module, masks: torch.BoolTensor, **kwargs) -> torch.Tensor:
        super().__init__()

        self.encoder = encoder

        self.register_buffer('masks', masks)

        self.heads = nn.ModuleList([
            MLP(input_size=int(size) + self.encoder.output_size, **kwargs)
            for size in self.masks.sum(dim=-1)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor, select: List[int] = None) -> torch.Tensor:
        z = self.encoder(x)

        if select is None:
            select = range(len(self.heads))

        fs = []
        for i, (mask, head) in enumerate(zip(self.masks, self.heads)):
            if i in select:
                fs.append(head(torch.cat([y[..., mask], z], dim=-1)))

        return torch.cat(fs, dim=-1)
