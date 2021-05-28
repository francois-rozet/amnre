#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchist

from typing import List, Union


Scalar = Union[bool, int, float]
Vector = Union[List[Scalar], np.ndarray, torch.Tensor]
Array = Union[List[List[Scalar]], np.ndarray, torch.Tensor]


def set_rcParams(usetex: bool = True):
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 12.
    plt.rcParams['legend.fontsize'] = 'small'
    plt.rcParams['savefig.transparent'] = True

    if usetex and mpl.checkdep_usetex(True):
        plt.rcParams['font.family'] = ['serif']
        plt.rcParams['font.serif'] = ['Computer Modern']
        plt.rcParams['text.usetex'] = True


def pairwise(hist: torch.Tensor) -> List[List[torch.Tensor]]:
    hists = []

    for i in range(hist.dim()):
        hists.append([])

        for j in range(i + 1):
            h = torchist.marginalize(hist, dim=[i, j], keep=True)

            if h.is_sparse:
                h = h.to_dense()

            hists[-1].append(h.cpu())

    return hists


def corner(
    hists: List[List[Array]],
    low: Vector,
    high: Vector,
    percentiles: Vector = [.1974, .3829, .6827, .8664, .9545, .9973],
    labels: List[str] = [],
    truth: Vector = None,
    filename: str = None,
    **fig_kwargs,
) -> mpl.figure.Figure:
    r"""Pairwise corner plot"""

    D = len(hists)

    fig_kwargs.setdefault('figsize', (D * 4.8,) * 2)
    fig, axs = plt.subplots(D, D, squeeze=False, **fig_kwargs)

    percentiles = np.sort(np.asarray(percentiles))
    percentiles = np.append(percentiles[::-1], 0.)

    for i in range(D):
        for j in range(D):
            ax = axs[i, j]

            # Only lower triangle
            if j > i or hists[i][j] is None:
                ax.axis('off')
                continue

            # Data
            hist = np.asarray(hists[i][j]).T
            x = np.linspace(low[j], high[j], hist.shape[-1])
            y = np.linspace(low[i], high[i], hist.shape[0])

            # Draw
            if i == j:
                ax.step(x, hist, color='k', linewidth=1.)
                ax.set_xlim(left=x[0], right=x[-1])
                ax.set_ylim(bottom=0.)
            else:
                levels = coverage(hist, percentiles)
                levels = np.unique(levels)

                cf = ax.contourf(
                    x, y, hist,
                    levels=levels,
                    cmap=NonLinearColormap('Blues', levels),
                    alpha=0.8,
                )
                ax.contour(cf, colors='k', linewidths=1.)

                if i > 0:
                    ax.sharex(axs[i - 1, j])

                if j > 0:
                    ax.sharey(axs[i, j - 1])

            ax.label_outer()
            ax.set_box_aspect(1.)

            # Labels
            if labels:
                if i == D - 1:
                    ax.set_xlabel(labels[j])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])

            # Truth
            if truth is not None:
                if i != j:
                    ax.plot(
                        truth[j], truth[i],
                        color='darkorange',
                        marker='o',
                        markersize=4.,
                    )
                else:
                    ax.axvline(
                        truth[i],
                        color='darkorange',
                        linewidth=2.,
                    )

    # Save file
    if filename is not None:
        plt.savefig(filename)
        plt.close()

        return None
    else:
        return fig


def coverage(x: np.ndarray, percentiles: Vector) -> np.ndarray:
    r"""Coverage percentiles"""

    x = np.sort(x, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, np.asarray(percentiles) * cdf[-1])

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