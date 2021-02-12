#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchist

from typing import Iterable, List, Union


Array = np.array
Tensor = torch.Tensor
ArrayLike = Union[List[int], List[float], Array]
TensorLike = Union[int, float, ArrayLike, Tensor]


plt.rcParams['axes.grid'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 12.
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True


def pairhist(
    samples: Iterable[Tensor],
    low: TensorLike,
    high: TensorLike,
    bins: Union[int, List[int]] = 100,
    normed: bool = False,
    bounded: bool = False,
) -> List[List[Array]]:
    r"""Pairwise histograms"""

    hists = None

    for x in samples:
        # Initialization
        if hists is None:
            D = x.size(-1)

            if type(bins) is int:
                bins = [bins] * D

            shapes = [[
                torch.Size((bins[i], bins[j]) if i != j else (bins[i],))
                for j in range(i + 1)
            ] for i in range(D)]

            hists = [[
                torch.zeros(shapes[i][j].numel()).to(x).long()
                for j in range(i + 1)
            ] for i in range(D)]

            bins = torch.tensor(bins).to(x)
            low, high = torch.tensor(low).to(x), torch.tensor(high).to(x)

        # Histograms
        if not bounded:
            mask = ~torchist.out_of_bounds(x, low, high)
            x = x[mask]

        x = torchist.discretize(x, bins, low, high).long()

        for i in range(D):
            for j in range(i + 1):
                if i != j:
                    indices = torchist.ravel_multi_index(x[..., [i, j]], shapes[i][j])
                else:
                    indices = x[..., i]

                hists[i][j] += indices.bincount(minlength=shapes[i][j].numel())

    for i in range(D):
        for j in range(i + 1):
            hists[i][j] = hists[i][j].float().view(shapes[i][j])

            if normed:
                hists[i][j] /= hists[i][j].sum()

            hists[i][j] = hists[i][j].cpu().numpy()

    return hists


def corner(
    hists: List[List[Array]],
    low: ArrayLike,
    high: ArrayLike,
    quantiles: List[float] = [.6827, .8664, .9545, .9876, .9973],
    labels: List[str] = [],
    **fig_kwargs,
) -> mpl.figure.Figure:
    r"""Pairwise corner plot"""

    D = len(hists)

    fig_kwargs.setdefault('figsize', (D * 4.8,) * 2)
    fig, axs = plt.subplots(D, D, squeeze=False, **fig_kwargs)

    for i in range(D):
        for j in range(D):
            if j > i:
                axs[i, j].axis('off')
                continue

            ax = axs[i, j]
            hist = hists[i][j]

            if i == j:
                x = np.linspace(low[i], high[i], hist.shape[0])

                ax.step(x, hist, color='k', linewidth=1.)
            else:
                x = np.linspace(low[j], high[j], hist.shape[1])
                y = np.linspace(low[i], high[i], hist.shape[0])

                levels = np.quantile(hist, quantiles + [1.])

                cf = ax.contourf(x, y, hist, levels=levels, cmap='Blues')
                ax.contour(cf, colors='k', linewidths=1.)

                if i > 0:
                    ax.sharex(axs[i - 1, j])

                if j > 0:
                    ax.sharey(axs[i, j - 1])

            ax.label_outer()
            ax.set_box_aspect(1.)

            if labels:
                if i == D - 1:
                    ax.set_xlabel(labels[j])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])

    return fig
