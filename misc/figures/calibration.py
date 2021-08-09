#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
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


if __name__ == '__main__':
    bins = 256

    ticks = np.linspace(0, 1, bins + 1)

    pdfs = {}
    pdfs['overdispersed'] = np.exp(-4 * (ticks - 0.75) ** 2)
    pdfs['underdispersed'] = 1.25 - np.exp(-4 * (ticks - 0.75) ** 2)
    pdfs['unsupported'] = np.tanh(8 * (0.5 - ticks)) + 1

    width, _ = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(width, width))

    for key, pdf in pdfs.items():
        cdf = np.cumsum(pdf)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

        plt.plot(ticks, cdf, label=key)

    plt.plot([0, 1], [0, 1], color='k', ls='--', label='calibrated')

    plt.grid()
    plt.xlabel(r'$p$')
    plt.ylabel(r'CDF$(p)$')
    plt.legend(loc='lower right')

    plt.savefig('calibration.pdf')
    plt.close()
