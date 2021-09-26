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
#SBATCH --time=24:00:00             # Max execution time
#
#SBATCH --array=1-3                 # Job array

conda activate amnre
cd ~/amnre

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"arch": "ResNet", "num_blocks": 17, "residual_size": 512, "activation": "ELU", "normalization": "batch"}'
FILTER='110111111001011'

python train.py -simulator GW -samples $SRC/gw-train.h5 -live -valid $SRC/gw-valid.h5 -model "$MODEL" -masks $FILTER -epochs 512 -descents 1024 -lr 2e-4 -scheduler cosine -o $DST/models/gw-nre.pth -device cuda

python eval.py $DST/models/gw-nre.pth $SRC/gw-event.h5 -masks $FILTER -o $DST/results/gw-nre-event.csv
