#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-amnre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-amnre.log       # Log-file
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi/python

python train.py -simulator HH -samples $SCRATCH/samples/hh-train.h5 -arbitrary -epochs 256 -o $SCRATCH/models/hh-amnre.pth -device cuda
python eval.py $SCRATCH/models/hh-amnre.json $SCRATCH/samples/hh-test.h5 -masks =8 -indices 0 3 -plots -o $SCRATCH/results/hh-amnre.csv
python eval.py $SCRATCH/models/hh-amnre.json $SCRATCH/samples/hh-test.h5 -masks =1 =2 -indices 0 3 -consistence -compositions 11111111 -o $SCRATCH/results/hh-amnre.csv
python eval.py $SCRATCH/models/hh-amnre.json $SCRATCH/samples/hh-test.h5 -masks =1 -indices 0 16384 -coverage -o $SCRATCH/results/hh-amnre_cov.csv
