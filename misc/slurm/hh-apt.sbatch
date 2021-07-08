#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-apt           # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-apt.log         # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=1-00:00:00           # Max execution time
#

conda activate amsi
cd ~/amsi

cp misc/references/apt_eval.py .
python apt_eval.py
rm apt_eval.py

mkdir -p $SCRATCH/references
mv hh-apt.* $SCRATCH/references
