#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=corners          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=corners.log        # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

mkdir -p $SCRATCH/corners
cp misc/corners/* $SCRATCH/corners
cd $SCRATCH

python ~/amsi/plots.py -corner corners/*.json
