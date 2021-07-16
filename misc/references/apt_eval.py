#!/usr/bin/env python

from amnre import HH
from delfi.inference import SNPEC
from delfi.utils import io

simulator = HH()

snpe = SNPEC(
    simulator.generator,
    obs=simulator.x_star[0],
    prior_norm=True,
    pilot_samples=1000,
    ## Network
    density='maf',
    n_mades=5,
    n_hiddens=[50, 50],
    impute_missing=False,
)

_, _, posterior = snpe.run(
    n_train=100000,
    n_rounds=1,
    proposal='mog',
    epochs=1000,
    minibatch=500,
    silent_fail=False,
    monitor_every=1,
    val_frac=0.1,
)
posterior = posterior[0]

io.save_pkl(posterior, 'hh-apt.pkl')

samples = posterior.gen(int(1e6))

import torch
import torchist

samples = torch.from_numpy(samples)

hist = torchist.histogramdd(samples, 100, simulator.low, simulator.high, sparse=True)
hist, _ = torchist.normalize(hist)

mask = torch.tensor([True] * hist.dim())
torch.save((mask, hist), 'hh-apt.pth')
