#!/usr/bin/env python

import bilby

trigger_time = 1126259462.391

priors = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
priors['geocent_time'].minimum = -.1
priors['geocent_time'].maximum = .1

result = bilby.result.read_in_result(outdir='GW150914', label='GW150914')
result.plot_corner()
result.plot_marginals()

samples = result.posterior[list(priors.keys())].values
samples[:, 3] -= trigger_time

import torch
import torchist

samples = torch.from_numpy(samples)

low = [p.minimum for p in priors.values()]
upp = [p.maximum for p in priors.values()]

hist = torchist.histogramdd(samples, 50, low, upp, bounded=True, sparse=True)
hist, _ = torchist.normalize(hist)

mask = torch.tensor([True] * hist.dim())
torch.save((mask, hist), 'gw-bilby.pth')
