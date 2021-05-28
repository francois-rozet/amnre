#!/usr/bin/env python

import torch
import torch.nn as nn

from typing import Callable, Iterable, Tuple, Union


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
}


class UnitNorm(nn.Module):
    r"""Unit Normalization (UnitNorm) layer

    Args:
        mu: The input mean
        sigma: The input standard deviation
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super().__init__()

        self.register_buffer('mu', mu)
        self.register_buffer('isigma', 1 / sigma)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mu) * self.isigma

    def extra_repr(self) -> str:
        mu, sigma = self.mu.cpu(), (1 / self.isigma).cpu()
        return '\n'.join([f'(mu): {mu}', f'(sigma): {sigma}'])


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
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.,
        activation: str = 'ReLU',
    ):
        dropout = nn.Dropout(dropout) if dropout > 0. else None
        activation = ACTIVATIONS[activation]()

        layers = [dropout, nn.Linear(input_size, hidden_size, bias), activation]

        for _ in range(num_layers):
            layers.extend([dropout, nn.Linear(hidden_size, hidden_size, bias), activation])

        layers.extend([dropout, nn.Linear(hidden_size, output_size, bias)])

        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.input_size = input_size
        self.output_size = output_size


class ResBlock(MLP):
    r"""Residual Block (ResBlock)

    Args:
        input_size: The input (and output) size.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(self, input_size: int, **kwargs):
        super().__init__(input_size, input_size, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super().forward(input)


class ResNet(nn.Sequential):
    r"""Residual Network (ResNet)

    Args:
        input_size: The input size.
        output_size: The output size.
        res_size: The intermediate residual size.
        num_blocks: The number of residual blocks.

        **kwargs are transmitted to `ResBlock` and `MLP`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        res_size: int = 64,
        num_blocks: int = 3,
        **kwargs,
    ):
        activation = ACTIVATIONS[kwargs.get('activation', 'ReLU')]
        bias = kwargs.get('bias', True)

        blocks = [nn.Linear(input_size, res_size, bias)]

        for _ in range(num_blocks):
            blocks.extend(ResBlock(res_size, **kwargs))

        blocks.append(nn.Linear(res_size, output_size, bias))

        super().__init__(*blocks)

        self.input_size = input_size
        self.output_size = output_size


def reparametrize(model: nn.Module, weights: torch.Tensor, i: int = 0) -> int:
    r"""Reparametrize model"""

    for key, val in model._parameters.items():
        j = i + val.numel()
        model._parameters[key] = weights[i:j].view(val.shape)
        i = j

    for m in model._modules.values():
        i = reparametrize(m, weights, i)

    return i


class HyperNet(nn.Module):
    r"""Hyper Network"""

    def __init__(self, network: nn.Module, input_size: int, **kwargs):
        super().__init__()

        output_size = sum(p.numel() for p in network.parameters())
        self.weights = MLP(input_size, output_size, **kwargs)

    def forward(self, network: nn.Module, condition: torch.Tensor) -> nn.Module:
        reparametrize(network, self.weights(condition))
        return network


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
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.normalize = nn.Identity() if moments is None else UnitNorm(*moments)
        self.mlp = MLP(theta_size + x_size, 1, **kwargs)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        theta = self.normalize(theta)
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
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        super().__init__()

        self.register_buffer('masks', masks)

        self.encoder = encoder

        if moments is not None:
            shift, scale = moments

        self.nres = nn.ModuleList([
            NRE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (shift[m], scale[m]),
                **kwargs,
            ) for m in self.masks
        ])

    def __getitem__(self, mask: torch.BoolTensor) -> nn.Module:
        mask = mask.to(self.masks)
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
        ratios = []
        for mask, nre in iter(self):
            ratios.append(nre(theta[..., mask], x))

        return torch.stack(ratios)


class AMNRE(nn.Module):
    r"""Arbitrary Marginal Neural Ratio Estimator (AMNRE)

    (theta, x, mask_a) ---> log r(theta_a | x)

    Args:
        theta_size: The size of the parameters.

        **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        encoder: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        hyper: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.normalize = nn.Identity() if moments is None else UnitNorm(*moments)

        if hyper is None:
            self.net = NRE(theta_size * 2, x_size, **kwargs)
            self.hyper = None
        else:
            self.net = NRE(theta_size, x_size, **kwargs)
            self.hyper = HyperNet(self.net, theta_size, **hyper)

        self.register_buffer('default', torch.ones(theta_size).bool())

    def clear(self) -> None:
        with torch.no_grad():
            self[torch.ones_like(self.default)]

    def __getitem__(self, mask: torch.BoolTensor) -> nn.Module:
        self.default = mask.to(self.default)

        if self.hyper is not None:
            self.hyper(self.net, self.default.float())

        return self

    def forward(
        self,
        theta: torch.Tensor,  # (N, D)
        x: torch.Tensor,  # (N, *)
        mask: torch.BoolTensor = None,  # (D,)
    ) -> torch.Tensor:
        if mask is None:
            mask = self.default
        elif self.hyper is not None:
            self.hyper(self.net, mask.float())

        if mask.dim() == 1 and theta.size(-1) < mask.numel():
            blank = theta.new_empty(theta.shape[:-1] + mask.shape)
            blank[..., mask] = theta
            theta = blank

        theta = self.normalize(theta) * mask

        if self.hyper is None:
            theta = torch.cat(torch.broadcast_tensors(theta, mask), dim=-1)

        return self.net(theta, x)
