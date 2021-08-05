#!/usr/bin/env python

import torch
import torch.nn as nn

from nflows import distributions, flows, transforms, utils
from torch.distributions import Distribution
from typing import List, Tuple, Union


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
}

NORMALIZATIONS = {
    'batch': nn.BatchNorm1d,
    'group': nn.GroupNorm,
    'layer': nn.LayerNorm,
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


def lecun_init(model):
    r"""LeCun normal weight initialization"""

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.kaiming_normal_(param, nonlinearity='linear')
        else:
            nn.init.constant_(param, 0.)


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        input_size: The input size.
        output_size: The output size.
        num_layers: The number of layers.
        hidden_size: The size of hidden layers.
        bias: Whether to use bias or not.
        activation: The activation layer type.
        dropout: The dropout rate.
        normalization: The normalization layer type.
        linear_first: Whether the first layer is a linear transform or not.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bias: bool = True,
        activation: str = 'ReLU',
        dropout: float = 0.,
        normalization: str = None,
        linear_first: bool = True,
        **absorb,
    ):
        activation = ACTIVATIONS[activation]
        selfnorm = normalization == 'self'

        if dropout == 0.:
            dropout = None
        elif selfnorm:
            dropout = nn.AlphaDropout(dropout)
        else:
            dropout = nn.Dropout(dropout)

        normalization = NORMALIZATIONS.get(normalization, lambda x: None)

        layers = [
            nn.Linear(input_size, hidden_size, bias) if linear_first else None,
            normalization(hidden_size),
            activation(),
        ]

        for _ in range(num_layers):
            layers.extend([
                dropout,
                nn.Linear(hidden_size, hidden_size, bias),
                normalization(hidden_size),
                activation(),
            ])

        layers.append(nn.Linear(hidden_size, output_size, bias))

        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        if selfnorm:
            lecun_init(self)

        self.input_size = input_size
        self.output_size = output_size


class ResBlock(MLP):
    r"""Residual Block (ResBlock)

    Args:
        size: The input, output and hidden sizes.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(self, size: int, **kwargs):
        super().__init__(size, size, size, linear_first=False, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super().forward(input)


class ResNet(nn.Sequential):
    r"""Residual Network (ResNet)

    Args:
        input_size: The input size.
        output_size: The output size.
        residual_size: The intermediate residual size.
        num_blocks: The number of residual blocks.

        **kwargs are transmitted to `ResBlock`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        residual_size: int = 64,
        num_blocks: int = 3,
        **kwargs,
    ):
        bias = kwargs.get('bias', True)

        blocks = [nn.Linear(input_size, residual_size, bias)]

        for _ in range(num_blocks):
            blocks.append(ResBlock(residual_size, **kwargs))

        blocks.append(nn.Linear(residual_size, output_size, bias))

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
    r"""Hyper Network (HyperNet)"""

    def __init__(self, network: nn.Module, input_size: int, **kwargs):
        super().__init__()

        output_size = sum(p.numel() for p in network.parameters())
        self.weights = ResNet(input_size, output_size, **kwargs)

    def forward(self, network: nn.Module, condition: torch.Tensor) -> nn.Module:
        reparametrize(network, self.weights(condition))
        return network


class NRE(nn.Module):
    r"""Neural Ratio Estimator (NRE)

    (theta, x) ---> log r(theta, x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the (encoded) observations.
        embedding: An optional embedding for the observations.

        **kwargs are transmitted to `MLP`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        embedding: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.embedding = embedding
        self.standardize = nn.Identity() if moments is None else UnitNorm(*moments)
        self.mlp = MLP(theta_size + x_size, 1, **kwargs)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        theta = self.standardize(theta)
        return self.mlp(torch.cat([theta, x], dim=-1)).squeeze(-1)


class MNRE(nn.Module):
    r"""Marginal Neural Ratio Estimator (MNRE)

                ---> log r(theta_a, x)
               /
    (theta, x) ----> log r(theta_b, x)
               \
                ---> log r(theta_c, x)

    Args:
        masks: The masks of the considered subsets of the parameters.
        x_size: The size of the (encoded) observations.
        embedding: An optional embedding for the observations.

        **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        masks: torch.BoolTensor,
        x_size: int,
        embedding: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer('masks', masks)
        self.embedding = embedding

        if moments is not None:
            shift, scale = moments

        self.nes = nn.ModuleList([
            NRE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (shift[m], scale[m]),
                **kwargs,
            ) for m in self.masks
        ])

    def __getitem__(self, mask: torch.BoolTensor) -> nn.Module:
        mask = mask.to(self.masks)

        for m, ne in iter(self):
            if (mask == m).all():
                return ne

        return None

    def __iter__(self): # -> Tuple[torch.BoolTensor, nn.Module]
        yield from zip(self.masks, self.nes)

    def filter(self, masks: torch.Tensor):
        nes = []

        for m in masks:
            nes.append(self[m])

        self.masks = masks
        self.nes = nn.ModuleList(nes)

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        preds = []

        for mask, ne in iter(self):
            preds.append(ne(theta[..., mask], x))

        return torch.stack(preds, dim=-1)


class AMNRE(nn.Module):
    r"""Arbitrary Marginal Neural Ratio Estimator (AMNRE)

    (theta, x, mask_a) ---> log r(theta_a, x)

    Args:
        theta_size: The size of the parameters.

        **kwargs are transmitted to `NRE`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        embedding: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        hyper: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.embedding = embedding
        self.standardize = nn.Identity() if moments is None else UnitNorm(*moments)

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
            self.hyper(self.net, self.default * 2. - 1.)

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
            self.hyper(self.net, mask * 2. - 1.)

        if mask.dim() == 1 and theta.size(-1) < mask.numel():
            blank = theta.new_zeros(theta.shape[:-1] + mask.shape)
            blank[..., mask] = theta
            theta = blank
        elif mask.dim() > 1 and theta.shape != mask.shape:
            batch_shape = theta.shape[:-1]
            stack_shape = batch_shape + mask.shape[:-1]
            view_shape = batch_shape + (1,) * (mask.dim() - 1)

            theta = theta.view(view_shape + theta.shape[-1:]).expand(stack_shape + theta.shape[-1:])
            x = x.view(view_shape + x.shape[-1:]).expand(stack_shape + x.shape[-1:])

        theta = self.standardize(theta) * mask

        if self.hyper is None:
            theta = torch.cat(torch.broadcast_tensors(theta, mask * 2. - 1.), dim=-1)

        return self.net(theta, x)


class MAF(flows.Flow):
    r"""Masked Autoregressive Flow (MAF)

    (x, y) -> log p(x | y)

    References:
        https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/flow.py
    """

    def __init__(
        self,
        x_size: int,
        y_size: int,
        num_transforms: int = 5,
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ):
        kwargs.setdefault('hidden_features', 64)
        kwargs.setdefault('num_blocks', 2)
        kwargs.setdefault('activation', torch.tanh)
        kwargs.setdefault('use_residual_blocks', False)
        kwargs.setdefault('use_batch_norm', True)

        transform = []

        if moments is not None:
            shift, scale = moments
            transform.append(
                transforms.PointwiseAffineTransform(-shift / scale, 1 / scale)
            )

        for _ in range(num_transforms if x_size > 1 else 1):
            transform.extend([
                transforms.MaskedAffineAutoregressiveTransform(
                    features=x_size,
                    context_features=y_size,
                    **kwargs
                ),
                transforms.RandomPermutation(features=x_size),
            ])

        transform = transforms.CompositeTransform(transform)
        distribution = distributions.StandardNormal((x_size,))

        super().__init__(transform, distribution)


class NSF(flows.Flow):
    r"""Neural Spline Flow (NSF)

    (x, y) -> log p(x | y)

    References:
        https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/flow.py
    """

    def __init__(
        self,
        x_size: int,
        y_size: int,
        num_transforms: int = 5,
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        others: dict = {'tails': 'linear'},
        **kwargs,
    ):
        kwargs.setdefault('hidden_features', 64)
        kwargs.setdefault('num_blocks', 2)
        kwargs.setdefault('use_residual_blocks', False)
        kwargs.setdefault('use_batch_norm', True)

        if kwargs['use_residual_blocks']:
            net = ResNet
            kwargs['residual_size'] = kwargs['hidden_features']
        else:
            net = MLP
            kwargs['hidden_size'] = kwargs['hidden_features']
            kwargs['num_layers'] = kwargs['num_blocks']

        if kwargs['use_batch_norm']:
            kwargs['normalization'] = 'batch'

        class ContextNet(nn.Module):
            def __init__(self, a: int, b: int):
                super().__init__()

                self.hidden_features = kwargs['hidden_features']
                self.net = net(a + y_size, b, **kwargs)

            def forward(self, input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
                return self.net(torch.cat([input, context], dim=-1))

        transform = []

        if moments is not None:
            shift, scale = moments
            transform.append(
                transforms.PointwiseAffineTransform(-shift / scale, 1 / scale)
            )

        for i in range(num_transforms if x_size > 1 else 1):
            transform.extend([
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=utils.create_alternating_binary_mask(x_size, even=(i % 2 == 0)),
                    transform_net_create_fn=ContextNet,
                    **others,
                ),
                transforms.LULinear(x_size, identity_init=True),
            ])

        transform = transforms.CompositeTransform(transform)
        distribution = distributions.StandardNormal((x_size,))

        super().__init__(transform, distribution)


class NPE(nn.Module):
    r"""Neural Posterior Estimator (NPE)

    (theta, x) ---> log p(theta | x)
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        embedding: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        design: str = 'MAF',
        prior: Distribution = None,
        **kwargs,
    ):
        super().__init__()

        self.embedding = embedding

        if design == 'NSF':
            flow = NSF
        else:  # flow == 'MAF'
            flow = MAF

        self.flow = flow(theta_size, x_size, moments=moments, **kwargs)

        self.prior = prior
        self._ratio = False

    def ratio(self, mode: bool = True):
        self._ratio = mode

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        prob = self.flow.log_prob(theta, x)

        if self._ratio:
            prob = prob - self.prior.log_prob(theta)

        return prob

    def sample(self, x: torch.Tensor, shape: torch.Size = ()) -> torch.Tensor:
        r""" theta ~ p(theta | x) """

        size = torch.Size(shape).numel()

        theta = self.flow._sample(size, x)
        theta = theta.view(x.shape[:-1] + shape + theta.shape[-1:])

        return theta


class MNPE(MNRE):
    r"""Marginal Neural Posterior Estimator (MNPE)

                ---> log p(theta_a | x)
               /
    (theta, x) ----> log p(theta_b | x)
               \
                ---> log p(theta_c | x)
    """

    def __init__(
        self,
        masks: torch.BoolTensor,
        x_size: int,
        embedding: nn.Module = nn.Identity(),
        moments: Tuple[torch.Tensor, torch.Tensor] = None,
        priors: List[Distribution] = None,
        **kwargs,
    ):
        super().__init__(masks, x_size, embedding)

        priors = [None] * len(masks) if priors is None else priors

        if moments is not None:
            shift, scale = moments

        self.nes = nn.ModuleList([
            NPE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (shift[m], scale[m]),
                prior=prior,
                **kwargs,
            )
            for m, prior in zip(self.masks, priors)
        ])

    def ratio(self, mode: bool = True):
        for _, ne in iter(self):
            ne.ratio(mode)
