#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import Callable, Iterable, Tuple, Union

Shape = Union[int, Iterable[int]]


class AttentiveLinear(nn.Module):
    r"""Attentive Linear (AL)

    Args:
        in_features: The input size.
        out_features: The output size.
        bias: Whether to use bias or not.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bias = nn.Linear(in_features, out_features, bias)
        self.weight = nn.Linear(in_features, in_features * out_features, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias(input)
        weight = self.weight(input).view(bias.shape + (self.in_features,))

        return weight @ input + bias


class Flatten(nn.Module):
    r"""Flattening module

    Args:
        input_shape: The input shape.
    """

    def __init__(self, input_shape: Shape):
        super().__init__()

        if type(input_shape) is int:
            input_shape = (input_shape,)

        self.input_shape = torch.Size(input_shape)

    @property
    def output_size(self):
        return self.input_shape.numel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(-len(self.input_shape))


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        input_shape: The input shape.
        output_size: The output size.
        num_layers: The number of layers.
        hidden_size: The size of hidden layers.
        bias: Whether to use bias or not..
        dropout: The dropout rate.
        activation: A callable returning an activation layer.
    """

    def __init__(
        self,
        input_shape: Shape,
        output_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        bias: bool = True,
        dropout: float = 0.,
        activation: Callable[[int], nn.Module] = nn.PReLU,
    ):
        if dropout > 0.:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        layers = [Flatten(input_shape)]

        layers.extend([
            nn.Linear(layers[0].output_size, hidden_size, bias),
            activation(hidden_size),
            dropout,
        ])

        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size, bias),
                activation(hidden_size),
                dropout,
            ])

        layers.append(nn.Linear(hidden_size, output_size, bias))

        super().__init__(*layers)

        self.input_shape = input_shape
        self.output_size = output_size


class NRE(MLP):
    r"""Neural Ratio Estimator (NRE)

    (theta, x) ---> log r(theta | x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the observations.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        **kwargs,
    ):
        super().__init__(theta_size + x_size, 1, **kwargs)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat([theta, x], dim=-1)).squeeze(-1)


class MNRE(nn.Module):
    r"""Marginal Neural Ratio Estimator (MNRE)

                ---> log r(theta_a | x)
               /
    (theta, x) ----> log r(theta_b | x)
               \
                ---> log r(theta_c | x)

    Args:
        masks: The masks of the considered subsets of the parameters.
        encoder: An optional encoder for the observations.
        x_size: The size of the (encoded) observations.

        **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        masks: torch.BoolTensor,
        encoder: nn.Module = nn.Identity(),
        x_size: int = None,
        **kwargs,
    ) -> torch.Tensor:
        super().__init__()

        self.register_buffer('masks', masks)

        self.encoder = encoder

        if hasattr(encoder, 'output_size'):
            x_size = encoder.output_size

        self.nres = nn.ModuleList([
            NRE(theta_size, x_size, **kwargs)
            for theta_size in self.masks.sum(dim=-1).tolist()
        ])

    def __len__(self):
        return len(self.nres)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, nn.Module]:
        return self.masks[i], self.nres[i]

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(x)

        ratios = []
        for mask, nre in zip(self.masks, self.nres):
            ratios.append(nre(theta[..., mask], x))

        return torch.stack(ratios, dim=-1)
