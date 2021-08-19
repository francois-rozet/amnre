#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'axes.titlesize': 'medium',
    'figure.autolayout': True,
    'figure.figsize': (3.2, 2.4),
    'font.size': 12.,
    'legend.fontsize': 'x-small',
    'lines.linewidth': 0.8,
    'lines.markersize': 6.,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'xtick.labelsize': 'x-small',
    'xtick.major.width': 0.8,
    'ytick.labelsize': 'x-small',
    'ytick.major.width': 0.8,
})

if mpl.checkdep_usetex(True):
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })


def nll(x: float, gamma: int = 2) -> float:
    return -np.log(x)


def fl(x: float, gamma: int = 2) -> float:
    return (1 - x) ** gamma * nll(x)


def pl(x: float, gamma: int = 2) -> float:
    return (1 - x ** gamma) * nll(x)


if __name__ == '__main__':
    ticks = 256
    x = np.linspace(0, 1, ticks + 1)[:-1] + 0.5 / ticks

    width, _ = plt.rcParams['figure.figsize']
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * width, width))

    for gamma in [1, 2, 3]:
        axs[0].plot(x, fl(x, gamma), label=rf'$\gamma = {gamma}$')
        axs[1].plot(x, pl(x, gamma), label=rf'$\gamma = {gamma}$')

    axs[0].set_title('FL')
    axs[1].set_title('PL')

    for ax in axs:
        ax.plot(x, nll(x), color='k', ls='--', label='NLL')
        ax.grid()
        ax.set_xlabel(r'$p$')
        ax.set_ylabel(r'$\mathcal{L}(p)$')
        ax.set_ylim(top=3.2, bottom=-0.2)

    ax.legend(loc='upper right')
    ax.label_outer()

    plt.savefig('flpl.pdf')
    plt.close()
