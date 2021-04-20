#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-amnre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-amnre.log       # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python train.py -simulator GW -samples $SCRATCH/samples/gw-train.h5 -arbitrary -epochs 256 -o $SCRATCH/models/gw-amnre.pth -device cuda
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =14 -indices 0 3 -plots -o $SCRATCH/results/gw-amnre.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =1 =2 -indices 0 3 -consistence -composition -o $SCRATCH/results/gw-amnre.csv
python eval.py $SCRATCH/models/gw-amnre.json $SCRATCH/samples/gw-test.h5 -masks =1 -indices 0 16384 -coverage -o $SCRATCH/results/gw-amnre_cov.csv
