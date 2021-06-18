#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-amnre       # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-amnre.log     # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator SLCP -samples $SCRATCH/samples/slcp-train.h5 -valid $SCRATCH/samples/slcp-valid.h5 -model "$MODEL" -arbitrary -o $SCRATCH/models/slcp-amnre.pth -device cuda

python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 =5 -indices 0 3 -o $SCRATCH/results/slcp-amnre.csv
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $SCRATCH/results/slcp-amnre_acc.csv
python eval.py $SCRATCH/models/slcp-amnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 -indices 0 16384 -clean -coverage -o $SCRATCH/results/slcp-amnre_cov.csv

python plots.py -loss $SCRATCH/models/slcp-amnre.csv
python plots.py -accuracy $SCRATCH/results/slcp-amnre_acc.csv
python plots.py -consistency $SCRATCH/results/slcp-amnre_acc_*.csv
python plots.py -coverage $SCRATCH/results/slcp-amnre_cov.csv
