#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchist

from typing import Iterable, List, Union


Number = Union[int, float]
ArrayLike = Union[Number, List[Number], np.ndarray, torch.Tensor]


plt.rcParams['axes.grid'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 12.
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True


def pairhist(
    samples: Iterable[torch.Tensor],
    low: ArrayLike,
    high: ArrayLike,
    bins: Union[int, List[int]] = 100,
    normed: bool = False,
    bounded: bool = False,
) -> List[List[torch.Tensor]]:
    r"""Pairwise histograms"""

    hists = None

    for x in samples:
        # Initialization
        if hists is None:
            D = x.size(-1)

            if type(bins) is int:
                bins = [bins] * D

            ## Shape of each histogram
            shapes = [[
                torch.Size((bins[i], bins[j]) if i != j else (bins[i],))
                for j in range(i + 1)
            ] for i in range(D)]

            ## Initialize at 0
            hists = [[
                torch.zeros(shapes[i][j].numel()).to(x)
                for j in range(i + 1)
            ] for i in range(D)]

            bins = torch.tensor(bins).to(x)

            if not torch.is_tensor(low):
                low, high = torch.tensor(low), torch.tensor(high)

            low, high = low.to(x), high.to(x)

        # Histograms

        ## Filter out-of-bounds
        if not bounded:
            mask = ~torchist.out_of_bounds(x, low, high)
            x = x[mask]

        ## Count
        x = torchist.discretize(x, bins, low, high).long()

        for i in range(D):
            for j in range(i + 1):
                if i != j:
                    indices = torchist.ravel_multi_index(x[..., [i, j]], shapes[i][j])
                else:
                    indices = x[..., i]

                hists[i][j] += indices.bincount(minlength=shapes[i][j].numel()).float()

    # Post-processing
    for i in range(D):
        for j in range(i + 1):
            if normed:
                hists[i][j] /= hists[i][j].sum()

            hists[i][j] = hists[i][j].cpu().view(shapes[i][j])

    return hists


def corner(
    hists: List[List[ArrayLike]],
    low: ArrayLike,
    high: ArrayLike,
    percentiles: ArrayLike = [.1974, .3829, .6827, .8664, .9545, .9973],
    labels: List[str] = [],
    **fig_kwargs,
) -> mpl.figure.Figure:
    r"""Pairwise corner plot"""

    D = len(hists)

    fig_kwargs.setdefault('figsize', (D * 4.8,) * 2)
    fig, axs = plt.subplots(D, D, squeeze=False, **fig_kwargs)

    low, high = np.asarray(low), np.asarray(high)
    percentiles = np.sort(np.asarray(percentiles))
    percentiles = np.append(percentiles[::-1], 0.)

    for i in range(D):
        for j in range(D):
            if j > i:
                axs[i, j].axis('off')
                continue

            ax = axs[i, j]
            hist = np.asarray(hists[i][j])

            if i == j:
                x = np.linspace(low[i], high[i], hist.shape[0])

                ax.step(x, hist, color='k', linewidth=1.)
            else:
                x = np.linspace(low[j], high[j], hist.shape[1])
                y = np.linspace(low[i], high[i], hist.shape[0])

                levels = coverage(hist, percentiles)

                cf = ax.contourf(
                    x, y, hist,
                    levels=levels,
                    cmap=NonLinearColormap('Blues', levels)
                )
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


def coverage(x: np.ndarray, percentiles: ArrayLike) -> np.ndarray:
    r"""Coverage percentiles"""

    x = np.sort(x, axis=None)[::-1]
    cdf = np.cumsum(x) / x.sum()
    idx = np.searchsorted(cdf, percentiles)

    return x[idx]


class NonLinearColormap(mpl.colors.LinearSegmentedColormap):
    r"""Non-linear colormap"""

    def __init__(self, cmap: str, levels: np.ndarray):
        self.cmap = plt.get_cmap(cmap)

        self.dom = (levels - levels.min()) / (levels.max() - levels.min())
        self.img = np.linspace(0., 1., len(levels))

    def __getattr__(self, attr: str):
        return getattr(self.cmap, attr)

    def __call__(self, x: np.ndarray, alpha: float = 1.0, **kwargs) -> np.ndarray:
        y = np.interp(x, self.dom, self.img)
        return self.cmap(y, alpha)
