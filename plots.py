#!/usr/bin/env python

import glob
import h5py
import itertools
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchist

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Union

import amsi


Array = np.ndarray
Scalar = Union[bool, int, float]
Vector = Union[List[Scalar], Array]


plt.rcParams.update({
    'axes.axisbelow': True,
    'axes.grid': True,
    'axes.linewidth': 1.,
    'figure.autolayout': True,
    'figure.figsize': (4., 3.),
    'font.size': 12.,
    'legend.fontsize': 'small',
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
})

if mpl.checkdep_usetex(True):
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })


#########
# Plots #
#########

def loss_plot(filename: str) -> mpl.figure.Figure:
    df = pd.read_csv(filename)

    fig = plt.figure()

    plt.plot(df['epoch'], df['mean'], label='training')
    plt.fill_between(df['epoch'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.25)

    plt.plot(df['epoch'], df['v_mean'], label='validation')
    plt.fill_between(df['epoch'], df['v_mean'] - df['v_std'], df['v_mean'] + df['v_std'], alpha=0.25)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    return fig


def bar_plot(df: pd.DataFrame, labels: List[str] = None, quantity: str = None) -> mpl.figure.Figure:
    df = df.groupby('mask', sort=False)
    mu, sigma = df.mean(), df.std()

    rows, cols = mu.shape
    width, height = plt.rcParams['figure.figsize']

    fig = plt.figure()

    ax = mu.plot.barh(
        width=.85, alpha=.66,
        xerr=sigma, error_kw={'elinewidth': 1.},
        figsize=(width, height * rows / 15)
    )

    ax.yaxis.grid(False)

    if labels is None:
        ax.get_legend().remove()
    else:
        ax.legend(labels)

    plt.xlabel(quantity)
    plt.ylabel(None)

    return fig


def coverage_plot(filename: str, bins: int = 20) -> mpl.figure.Figure:
    df = pd.read_csv(filename, usecols=['index', 'mask', 'quantile'], dtype={'mask': str})

    hists = []

    for mask in set(df['mask']):
        q = df['quantile'][df['mask'] == mask]

        hist, _ = np.histogram(q, bins=bins, range=(0., 1.))
        hist = hist / len(q)

        hists.append(hist)

    hists = np.stack(hists)

    ticks = np.linspace(0., 1., bins + 1)
    mu = hists.mean(axis=0)
    sigma = hists.std(axis=0)

    ticks = np.repeat(ticks, 2)
    mu = np.hstack([0., np.repeat(mu, 2), 0.])
    sigma = np.hstack([0., np.repeat(sigma, 2), 0.])

    fig = plt.figure()

    plt.plot(ticks, mu)
    plt.fill_between(ticks, mu - sigma, mu + sigma, alpha=0.25)

    plt.axhline(y=1 / bins, color='tab:orange', ls='--')

    plt.xlabel('Percentile')
    plt.ylabel('Frequency')

    return fig


def visible_spines(ax: mpl.axes.Axes = None) -> None:
    if ax is None:
        ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_visible(True)


def consistency_plot(
    files: List[str],
    cmap: str = 'Reds',
    background: tuple = (.9, .9, .9),
) -> None:
    a, b = files[:2]
    for i in range(len(a)):
        if a[i] != b[i]:
            break

    basename = a[:i]

    dfs = []

    for f in files:
        df = pd.read_csv(f, dtype={0: str})
        df = df.set_index(df.columns[0])

        dfs.append(df.values)

    dfs = np.stack(dfs)
    labels = df.columns

    mu = dfs.mean(axis=0)
    sigma = dfs.std(axis=0)

    vmax = max(np.nanmax(mu), np.nanmax(sigma))

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(background)

    rows, cols = mu.shape
    width, height = plt.rcParams['figure.figsize']

    for i, mat in enumerate([mu, sigma]):
        fig = plt.figure(figsize=(width * rows / 15, height * rows / 15))

        ax = sns.heatmap(
            mat,
            cmap=cmap,
            square=True,
            vmin=0., vmax=vmax,
            xticklabels=labels, yticklabels=labels,
            cbar=i == 1,
            cbar_kws={'pad': 0.033},
        )

        visible_spines(ax)

        if i == 1:
            visible_spines(ax.collections[-1].colorbar.ax)
            ax.set_yticks([])

        plt.savefig(basename + ('mean' if i == 0 else 'std') + '.pdf')
        plt.close()

    return fig


def roc_plot(filename: str) -> mpl.figure.Figure:  # Receiver Operating Characteristic
    with h5py.File(filename) as f:
        for mask, data in f.items():
            fp, tp, _ = roc_curve(data[:, 0], data[:, 1], sample_weight=data[:, 2])
            area = round(auc(fp, tp), 3)

            plt.figure()

            plt.step(fp, tp, label=f'ROC (AUC = {area})')
            plt.plot([0, 1], [0, 1], linestyle='--')

            plt.axis('square')

            plt.xlabel('False Positive rate')
            plt.ylabel('True Positive rate')
            plt.legend()

            plt.savefig(filename.replace('.h5', f'_{mask}.pdf'))
            plt.close()


##########
# Corner #
##########

SIMULATORS = {
    'SLCP': amsi.SLCP,
    'MLCP': amsi.MLCP,
    'GW': amsi.GW,
    'HH': amsi.HH,
}


def pairwise_marginalize(hist: torch.Tensor, dims: List[int]) -> Dict[Tuple[int, int], Array]:
    pairs = {}

    for i, a in enumerate(dims):
        for j, b in enumerate(dims[:i + 1]):
            h = torchist.marginalize(hist, dim=[i, j], keep=True)

            if h.is_sparse:
                h = h.to_dense()

            pairs[(a, b)] = h.t().cpu().numpy()

    return pairs


def search_quantiles(x: Array, quantiles: Vector) -> Vector:
    x = np.sort(x, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, np.asarray(quantiles) * cdf[-1])

    return x[idx]


class AlphaLinearColormap(mpl.colors.Colormap):
    def __new__(self, color: str, alpha: float = 1.):
        rgb = mpl.colors.to_rgb(color)

        return mpl.colors.LinearSegmentedColormap.from_list(
            color + '_v',
            (rgb + (0.,), rgb + (alpha,))
        )


class NonLinearColormap(mpl.colors.Colormap):
    def __init__(self, cmap: mpl.colors.Colormap, levels: Vector):
        super().__init__(name=cmap.name + '_nl')

        self.cmap = cmap

        levels = np.asarray(levels)

        self.dom = (levels - levels.min()) / (levels.max() - levels.min())
        self.img = np.linspace(0., 1., len(levels))

    def __getattr__(self, attr: str):
        return getattr(self.cmap, attr)

    def __call__(self, x: np.ndarray, alpha: float = 1.0, **kwargs) -> np.ndarray:
        y = np.interp(x, self.dom, self.img)
        return self.cmap(y, alpha)


def corner(
    data: Dict[str, Dict[Tuple[int, int], Array]],
    dims: List[int],
    low: Vector,
    high: Vector,
    quantiles: Vector = [.3829, .6827, .9545, .9973],
    labels: List[str] = None,
    truth: Vector = None,
    legend: bool = False,
) -> mpl.figure.Figure:
    D = len(dims)

    fontsize = D * plt.rcParams['font.size'] / 3
    squaresize = plt.rcParams['figure.figsize'][1]

    fig, axs = plt.subplots(
        D, D,
        squeeze=False,
        figsize=(D * squaresize,) * 2,
    )

    quantiles = np.sort(np.asarray(quantiles))
    quantiles = np.append(quantiles[::-1], 0.)

    colors = {key: c for (key, c) in zip(data, mpl.colors.TABLEAU_COLORS)}

    for i in range(D):
        for j in range(D):
            ax = axs[i, j]

            # Only lower triangle
            if j > i:
                ax.axis('off')
                continue

            # Data
            a, b = dims[i], dims[j]

            for key, hists in data.items():
                if (a, b) in hists:
                    hist = hists[(a, b)]
                else:
                    continue

                x = np.linspace(low[b], high[b], hist.shape[-1])
                y = np.linspace(low[a], high[a], hist.shape[0])

                ## Color
                color = colors[key]
                cmap = AlphaLinearColormap(color, 0.5)

                ## Draw
                if i == j:
                    ax.step(x, hist, color=color)

                    _, ymax = ax.get_ylim()
                    ymax = max(ymax, hist.max() * 1.0625)

                    ax.set_xlim(left=x[0], right=x[-1])
                    ax.set_ylim(bottom=0., top=ymax)
                else:
                    levels = search_quantiles(hist, quantiles)
                    levels = np.unique(levels)

                    cf = ax.contourf(
                        x, y, hist,
                        levels=levels,
                        cmap=NonLinearColormap(cmap, levels),
                    )
                    ax.contour(cf, colors=color)

                    if i > 0:
                        ax.sharex(axs[i - 1, j])

                    if j > 0:
                        ax.sharey(axs[i, j - 1])

            ax.label_outer()
            ax.set_box_aspect(1.)

            # Ticks
            ax.tick_params(
                bottom=i == D - 1,
                left=j == 0,
            )

            # Labels
            if labels is not None:
                if i == D - 1:
                    ax.set_xlabel(labels[b], fontsize=fontsize)

                if j == 0 and i != j:
                    ax.set_ylabel(labels[a], fontsize=fontsize)

            # Truth
            if truth is not None:
                if i != j:
                    ax.plot(
                        truth[b], truth[a],
                        color='k',
                        marker='*',
                        markersize=9.,
                    )
                else:
                    ax.axvline(
                        truth[a],
                        color='k',
                        ls='--',
                    )

    if legend:
        labels = [key for key in colors if not key.startswith('_')]
        handles = [
            mpl.lines.Line2D([0], [0], color=colors[key], linewidth=D / 2)
            for key in labels
        ]

        fig.legend(handles, labels, loc='upper right', fontsize=fontsize)

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Display results')

    parser.add_argument('-loss', nargs='+', default=[], help='training loss file (CSV)')
    parser.add_argument('-accuracy', nargs='+', default=[], help='accuracy file (CSV)')
    parser.add_argument('-coverage', nargs='+', default=[], help='coverage file (CSV)')
    parser.add_argument('-consistency', nargs='+', default=[], help='consistency files (CSV)')
    parser.add_argument('-roc', nargs='+', default=[], help='prediction file (H5)')
    parser.add_argument('-corner', nargs='+', default=[], help='corner settings file (JSON)')

    parser.add_argument('-metrics', nargs='+',
        default=['probability', 'entropy', 'distance'],
        choices=['probability', 'entropy', 'distance'],
        help='accuracy metrics'
    )

    args = parser.parse_args()

    # Loss plots
    for filename in args.loss:
        loss_plot(filename)

        plt.savefig(filename.replace('.csv', '.pdf'))
        plt.close()

    # Accuracy plots
    for filename in args.accuracy:
        df = pd.read_csv(filename, dtype={'mask': str})

        for metric in args.metrics:
            if metric == 'probability':
                temp = df[['mask', 'total_probability']]
                labels = None
                quantity = 'Total probability'
            elif metric == 'entropy':
                temp = df[['mask', 'entropy', 'entropy_truth']]
                labels = ['prediction', 'ground truth']
                quantity='Entropy'
            elif metric == 'distance':
                temp = df[['mask', 'wd_truth']]
                labels = None
                quantity=r'$W_d$ to ground truth'

            bar_plot(temp, labels, quantity)

            plt.savefig(filename.replace('.csv', f'_{metric}.pdf'))
            plt.close()

    # Coverage plots
    for filename in args.coverage:
        coverage_plot(filename)

        plt.savefig(filename.replace('.csv', '.pdf'))
        plt.close()

    # Consistency plots
    if len(args.consistency) > 1:
        consistency_plot(args.consistency)

    # ROC plots
    for filename in args.roc:
        roc_plot(filename)

    # Corner plots
    for filename in args.corner:
        with open(filename) as g:
            settings = json.load(g)

        ## Simulator
        simulator = SIMULATORS[settings['simulator']]()

        labels = simulator.labels
        low = simulator.low.numpy()
        high = simulator.high.numpy()

        ## Star
        if 'star' in settings:
            samples, index = settings['star']

            with h5py.File(samples) as f:
                theta_star = f['theta'][index]
        else:
            theta_star = None

        ## Histograms
        data = {}

        for key, files in settings['items'].items():
            files = [g for f in files for g in glob.glob(f)]

            if len(files) == 1:
                mask, hist = torch.load(files[0])
                dims = mask.nonzero().squeeze(-1).tolist()

                data[key] = pairwise_marginalize(hist, dims)
            else:
                data[key] = {}

                for f in files:
                    mask, hist = torch.load(f)
                    dims = mask.nonzero().squeeze(-1).tolist()

                    if len(dims) == 1:
                        i = j = dims[0]
                    elif len(dims) == 2:
                        j, i = dims
                    else:
                        continue

                    data[key][(i, j)] = hist.t().numpy()

        ## Masks
        for textmask in settings['masks']:
            dims = amsi.str2mask(textmask).nonzero().squeeze(-1).tolist()

            corner(
                data, dims, low, high,
                labels=labels, truth=theta_star,
                legend=settings.get('legend', False),
            )

            plt.savefig(filename.replace('.json', f'_{textmask}.pdf'))
            plt.close()
