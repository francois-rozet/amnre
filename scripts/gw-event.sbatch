#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-event         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-event.log       # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python

python samples.py -simulator GW -seed 0 -events -o $SCRATCH/samples/gw-event.h5
