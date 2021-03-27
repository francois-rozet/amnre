#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp             # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp.log           # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=00:10:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python samples.py -simulator SLCP -seed 0 -o $SCRATCH/samples/slcp.h5
