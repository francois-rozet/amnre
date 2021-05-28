#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-amnre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-amnre.log       # Log-file
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

export GWPY_RCPARAMS=0

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator GW -samples $SCRATCH/samples/gw-train.h5 -valid $SCRATCH/samples/gw-valid.h5 -model "$MODEL" -arbitrary -o $SCRATCH/models/gw-amnre.pth -device cuda

python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-event.h5 -masks =15 -plots -o $SCRATCH/results/gw-amnre-event.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-event.h5 -masks =1 =2 -compositions 111111111111111 -o $SCRATCH/results/gw-amnre-event.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =15 -indices 0 3 -plots -o $SCRATCH/results/gw-amnre.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =1 =2 -indices 0 3 -consistence -compositions 111111111111111 -o $SCRATCH/results/gw-amnre.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =1 -indices 0 16384 -coverage -o $SCRATCH/results/gw-amnre_cov.csv
