#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-pamnre        # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-pamnre.log      # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

export GWPY_RCPARAMS=0

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator GW -samples $SCRATCH/samples/gw-train.h5 -valid $SCRATCH/samples/gw-valid.h5 -model "$MODEL" -masks poisson -arbitrary -o $SCRATCH/models/gw-pamnre.pth -device cuda

python eval.py $SCRATCH/models/gw-pamnre.json $SCRATCH/samples/gw-event.h5 -masks =1 =2 -compositions 111111111111111 -o $SCRATCH/results/gw-pamnre-event.csv
python eval.py $SCRATCH/models/gw-pamnre.json $SCRATCH/samples/gw-test.h5 -masks =1 =2 -indices 0 3 -consistence -compositions 111111111111111 -o $SCRATCH/results/gw-pamnre.csv
python eval.py $SCRATCH/models/gw-pamnre.json $SCRATCH/samples/gw-test.h5 -masks =1 -indices 0 16384 -coverage -o $SCRATCH/results/gw-pamnre_cov.csv
