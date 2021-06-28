#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-nre           # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-nre_%a.log      # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=12:00:00             # Max execution time
#
#SBATCH --array=1-3                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 11, "hidden_size": 512, "activation": "ELU"}'
ENCODER='{"name": "ResNet", "num_blocks": 10, "residual_size": 512, "output_size": 512, "activation": "ELU"}'

python train.py -simulator GW -samples $SRC/gw-train.h5 -valid $SRC/gw-valid.h5 -model "$MODEL" -encoder "$ENCODER" -epochs 512 -per-epoch 1024 -lr 2e-4 -patience 11 -o $DST/models/gw-nre.pth -device cuda

python eval.py $DST/models/gw-nre.pth $SRC/gw-event.h5 -o $DST/results/gw-nre-event.csv
