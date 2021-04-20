#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-nre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-nre.log       # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python train.py -simulator SLCP -samples $SCRATCH/samples/slcp-train.h5 -o $SCRATCH/models/slcp-nre.pth -device cuda
python eval.py $SCRATCH/models/slcp-nre.json $SCRATCH/samples/slcp-test.h5 -indices 0 3 -accuracy -plots -o $SCRATCH/results/slcp-nre.csv
