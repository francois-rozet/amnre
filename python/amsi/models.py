#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import Callable, Iterable, Tuple, Union

from .utils import enumerate_masks


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
}


class AttentiveLinear(nn.Module):
    r"""Attentive Linear layer

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, 1, shape[-1])

        b = self.bias(x)
        W = self.weight(x).view(-1, self.in_features, self.out_features)

        y = torch.bmm(x, W) + b
        y = y.view(shape[:-1] + (-1,))

        return y


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        input_size: The input size.
        output_size: The output size.
        num_layers: The number of layers.
        hidden_size: The size of hidden layers.
        bias: Whether to use bias or not.
        dropout: The dropout rate.
        activation: The activation layer type.
        attentive: Whether to use attentive linear layers or not.
    """

    def __init__(
        self,
        input_size: Union[int, torch.Size],
        output_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        bias: bool = True,
        dropout: float = 0.,
        activation: str = 'ReLU',
        attentive: bool = False,
    ):
        dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        activation = ACTIVATIONS[activation]()
        linear = AttentiveLinear if attentive else nn.Linear

        layers = []

        if type(input_size) is not int:
            layers.append(nn.Flatten(-len(input_size)))
            input_size = input_size.numel()

        layers.extend([
            linear(input_size, hidden_size, bias),
            activation,
            dropout,
        ])

        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size, bias),
                activation,
                dropout,
            ])

        layers.append(linear(hidden_size, output_size, bias))

        super().__init__(*layers)

        self.input_size = input_size
        self.output_size = output_size


class NRE(nn.Module):
    r"""Neural Ratio Estimator (NRE)

    (theta, x) ---> log r(theta | x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the (encoded) observations.
        encoder: An optional encoder for the observations.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        encoder: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__()

        self._encode = True
        self.encoder = encoder

        self.mlp = MLP(theta_size + x_size, 1, **kwargs)

    def set_encode(self, mode: bool):
        self._encode = mode

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self._encode:
            x = self.encoder(x)

        return self.mlp(torch.cat([theta, x], dim=-1)).squeeze(-1)


class MNRE(nn.Module):
    r"""Marginal Neural Ratio Estimator (MNRE)

                ---> log r(theta_a | x)
               /
    (theta, x) ----> log r(theta_b | x)
               \
                ---> log r(theta_c | x)

    Args:
        masks: The masks of the considered subsets of the parameters.
        x_size: The size of the (encoded) observations.
        encoder: An optional encoder for the observations.

        **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        masks: torch.BoolTensor,
        x_size: int,
        encoder: nn.Module = nn.Identity(),
        **kwargs,
    ) -> torch.Tensor:
        super().__init__()

        self.register_buffer('masks', masks)

        self._encode = True
        self.encoder = encoder

        self.nres = nn.ModuleList([
            NRE(theta_size, x_size, **kwargs)
            for theta_size in self.masks.sum(dim=-1).tolist()
        ])

    def set_encode(self, mode: bool):
        self._encode = mode

    def __getitem__(self, mask: torch.BoolTensor) -> nn.Module:
        match = torch.all(self.masks == mask, dim=-1)

        if torch.any(match):
            i = match.byte().argmax().item()
            return self.nres[i]
        else:
            return None

    def __iter__(self) -> Tuple[torch.BoolTensor, nn.Module]:
        yield from zip(self.masks, self.nres)

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self._encode:
            x = self.encoder(x)

        ratios = []
        for mask, nre in iter(self):
            ratios.append(nre(theta[..., mask], x))

        return torch.stack(ratios, dim=-1)


class AMNRE(NRE):
    r"""Arbitrary Marginal Neural Ratio Estimator (AMNRE)

    (theta, x, mask_a) ---> log r(theta_a | x)

    Args:
        theta_size: The size of the parameters.
        masks: The masks of the considered subsets of the parameters.
            Only used during training.

        *args and **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        theta_size: int,
        *args,
        masks: torch.BoolTensor = None,
        **kwargs,
    ):
        super().__init__(theta_size * 2, *args, **kwargs)

        if masks is None:
            masks = enumerate_masks(theta_size)

        self.register_buffer('masks', masks)
        self.register_buffer('default', torch.ones(theta_size).bool())

    def __getitem__(self, mask: torch.BoolTensor) -> nn.Module:
        with torch.no_grad():
            self.default.data = mask.to(self.default)

        return self

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        if mask is None:
            if self.training:
                idx = torch.randint(len(self.masks), theta.shape[:-1])
                mask = self.masks[idx]
            else:
                mask = self.default

        if mask.dim() == 1 and theta.size(-1) < mask.numel():
            blank = theta.new_zeros(theta.shape[:-1] + mask.shape)
            blank[..., mask] = theta
            theta = blank
        else:
            theta = theta * mask

        theta = torch.cat(torch.broadcast_tensors(theta, mask), dim=-1)

        return super().forward(theta, x)
