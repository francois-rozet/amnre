#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-amnre       # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-amnre.log     # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=01:00:00             # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python train.py -simulator SLCP -samples $SCRATCH/samples/slcp-train.h5 -arbitrary -o $SCRATCH/models/slcp-amnre.pth -device cuda
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =5 -indices 0 3 -plots -o $SCRATCH/results/slcp-amnre.csv
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 -indices 0 3 -consistence -composition -o $SCRATCH/results/slcp-amnre.csv
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 -indices 0 64 -accuracy -o $SCRATCH/results/slcp-amnre_acc.csv
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 -indices 0 16384 -coverage -o $SCRATCH/results/slcp-amnre_cov.csv
