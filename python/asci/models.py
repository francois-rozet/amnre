#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import Callable


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
        activation: Callable[[], nn.Module] = nn.PReLU,
    ):
        if dropout > 0.:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        layers = [nn.Linear(input_size, hidden_size, bias), activation(), dropout]

        for i in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size, bias), activation(), dropout])

        layers.append(nn.Linear(hidden_size, output_size, bias))

        super().__init__(*layers)
