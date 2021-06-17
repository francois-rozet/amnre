r"""Arbitrary Marginal Simulation-based Inference"""

from .criteria import WeightedLoss, RELoss, RDLoss, SDLoss
from .datasets import OnlineLTEDataset, OfflineLTEDataset
from .models import MLP, ResNet, NRE, MNRE, AMNRE
from .samplers import TractableSampler, RESampler

from .simulators import Simulator, LTERatio
from .simulators.slcp import SLCP, MLCP
from .simulators.gw import GW, svd_basis
from .simulators.hh import HH

from .masks import *
from .histograms import set_rcParams, pairwise, corner
