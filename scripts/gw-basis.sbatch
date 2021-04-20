#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-basis         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-basis.log       # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python samples.py -simulator GW -seed 0 -samples 1024 -o $SCRATCH/samples/gw-basis.h5
