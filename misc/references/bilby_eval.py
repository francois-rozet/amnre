#!/usr/bin/env python

import bilby

trigger_time = 1126259462.391

priors = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
priors['geocent_time'].minimum = -.1
priors['geocent_time'].maximum = .1

result = bilby.result.read_in_result(outdir='GW150914', label='GW150914')
result.plot_corner()
result.plot_marginals()

keys = [
    'mass_1', 'mass_2', 'phase', 'geocent_time', 'luminosity_distance',
    'a_1', 'a_2', 'tilt_1', 'tilt_2',
    'phi_12', 'phi_jl', 'theta_jn',
    'psi', 'ra', 'dec',
]

samples = result.posterior[keys].values
samples[:, 3] -= trigger_time

import torch
import torchist

samples = torch.from_numpy(samples)

low = [priors[key].minimum for key in keys]
upp = [priors[key].maximum for key in keys]

hist = torchist.histogramdd(samples, 50, low, upp, bounded=True, sparse=True)
hist, _ = torchist.normalize(hist)

mask = torch.tensor([True] * hist.dim())
torch.save((mask, hist), 'gw-bilby.pth')
