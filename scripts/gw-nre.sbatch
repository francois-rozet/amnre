#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-nre           # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-nre.log         # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python

export GWPY_RCPARAMS=0

python train.py -simulator GW -samples $SCRATCH/samples/gw-train.h5 -epochs 256 -o $SCRATCH/models/gw-nre.pth -device cuda
python eval.py $SCRATCH/models/gw-nre.json $SCRATCH/samples/gw-event.h5 -indices 0 1 -plots -o $SCRATCH/results/gw-nre-event.csv
python eval.py $SCRATCH/models/gw-nre.json $SCRATCH/samples/gw-test.h5 -indices 0 3 -plots -o $SCRATCH/results/gw-nre.csv
