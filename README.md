# Arbitrary Marginal Neural Ratio Estimation for Likelihood-free Inference

This repository is the official implementation of *Arbitrary Marginal Neural Ratio Estimation* (AMNRE) for *Likelihood-free Inference* (LFI).

## Code

The majority of the code is written in [Python](https://www.python.org/) with some `bash` automation. The neural networks are built and trained using the [PyTorch](https://pytorch.org/) automatic differentiation framework. We also rely on [`nflows`](https://github.com/bayesiains/nflows) to implement normalizing flow networks, [`torchist`](https://github.com/francois-rozet/torchist) to manipulate histograms and [`matplotlib`](https://github.com/matplotlib/matplotlib) to display results graphically.

All dependencies are provided as a `conda` [environment file](environment.yml), which can be installed with

```bash
conda env create -f environment.yml
conda activate amnre
```

### Organization

The core of the code is the [`amnre`](amnre/) folder, which is a Python package. It contains the implementation of the [models](amnre/models.py), [simulators](amnre/simulators/), [data loaders](amnre/datasets.py) and more.

### Simulators and datasets

Three simulators are provided: Simple Likelihood and Complex Posterior (`SLCP`), Hodgkin-Huxley (`HH`) and Gravitational Waves (`GW`). Other simulators can be implemented by extending the `Simulator` class. In order to use `HH` and `GW`, the code of other repositories has to imported and compiled, which is automated by [hh_setup.sh](misc/hh_setup.sh) and [gw_setup.sh](misc/gw_setup.sh), respectively.

```bash
conda activate amnre
bash misc/hh_setup.sh
bash misc/gw_setup.sh
```

> If you don't plan on using `HH` or `GW` this step can be skipped.

To store and access samples from these simulators, we use the `HDF5` file format with its Python interface [`h5py`](https://github.com/h5py/h5py). In particular, we store the parameters in a dataset `theta` and the realizations in a dataset `x` within a `.h5` file. From such file, an `OfflineDataset` acting as an iterable data loader can be instantiated. The latter has the advantage over `torch`'s builtin data loaders to only load a *chunk* of the data on RAM, at once, which is necessary when realizations are large. The script [`sample.py`](sample.py) generates samples from the simulators and creates the `.h5` files.

For example, to create training and validation datasets for the `SLCP` simulator, we use

```bash
python sample.py -simulator SLCP -seed 0 -samples 1048576 -o slcp-train.h5
python sample.py -simulator SLCP -seed 1 -samples 131072 -o slcp-valid.h5
```

### Training

The script [`train.py`](train.py) links the different modules of `amnre` to instantiate and train the models. A lot of options are available and can be listed with the `--help` flag.

For example, to train an `AMNRE` model with the `SLCP` training dataset, we use

```bash
MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'
python train.py -simulator SLCP -samples slcp-train.h5 -valid slcp-valid.h5 -model "$MODEL" -arbitrary -device cuda -o slcp-amnre.pth
```

The outputs are tree files: the instance settings `slcp-amnre.json`, the network weights `slcp-amnre.pth` and the training statistics `slcp-amnre.csv`. The latter reports, for each epoch, the time, the learning rate, the mean and standard deviation of the training loss, and the mean and standard deviation of the validation loss.

```csv
epoch,time,lr,mean,std,v_mean,v_std
1,2.0477960109710693,0.001,0.52534419298172,0.27149534225463867,0.23883646726608276,0.01787589117884636
2,1.9976160526275635,0.001,0.20120219886302948,0.04054423049092293,0.16070948541164398,0.016732219606637955
3,1.9926025867462158,0.001,0.13525991141796112,0.07273339480161667,0.08658836781978607,0.01457754522562027
...
```

> The settings and weights files should always stay together as the former contains the instructions to build the network.

### Evaluation

The evaluation process is done in two stages: [`eval.py`](eval.py) performs the actual computations and [`plots.py`](plots.py) represents graphically the results. These scripts are heavily specialized for our experiments. If you wish to perform other experiments, it is probably easier to write your own evaluation procedures.

### Experiments

Our experiments were performed on a cluster of GPUs managed with [Slurm](https://slurm.schedmd.com/documentation.html). All the scripts are provided in the [slurm](misc/slurm/) folder.

## References

```bib
@article{cranmer2020frontier,
  title={The frontier of simulation-based inference},
  author={Cranmer, Kyle and Brehmer, Johann and Louppe, Gilles},
  journal={Proceedings of the National Academy of Sciences},
  volume={117},
  number={48},
  pages={30055--30062},
  year={2020},
  publisher={National Acad Sciences}
}

@inproceedings{hermans2020likelihood,
  title={Likelihood-free mcmc with amortized approximate ratio estimators},
  author={Hermans, Joeri and Begy, Volodimir and Louppe, Gilles},
  booktitle={International Conference on Machine Learning},
  pages={4239--4248},
  year={2020},
  organization={PMLR}
}
```
