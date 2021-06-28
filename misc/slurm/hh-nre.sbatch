#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-nre           # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-nre_%a.log      # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'

python train.py -simulator HH -samples $SRC/hh-train.h5 -valid $SRC/hh-valid.h5 -model "$MODEL" -o $DST/models/hh-nre.pth -device cuda

python eval.py $DST/models/hh-nre.pth $SRC/hh-event.h5 -o $DST/results/hh-nre-event.csv
