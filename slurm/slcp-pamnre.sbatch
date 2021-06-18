#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-pamnre      # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-pamnre.log    # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator SLCP -samples $SCRATCH/samples/slcp-train.h5 -valid $SCRATCH/samples/slcp-valid.h5 -model "$MODEL" -masks poisson -arbitrary -o $SCRATCH/models/slcp-pamnre.pth -device cuda

python eval.py $SCRATCH/models/slcp-pamnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 =5 -indices 0 3 -o $SCRATCH/results/slcp-pamnre.csv
python eval.py $SCRATCH/models/slcp-pamnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $SCRATCH/results/slcp-pamnre_acc.csv
python eval.py $SCRATCH/models/slcp-pamnre.json $SCRATCH/samples/slcp-test.h5 -masks =1 -indices 0 16384 -clean -coverage -o $SCRATCH/results/slcp-pamnre_cov.csv

python plots.py -loss $SCRATCH/models/slcp-pamnre.csv
python plots.py -accuracy $SCRATCH/results/slcp-pamnre_acc.csv
python plots.py -consistency $SCRATCH/results/slcp-pamnre_acc_*.csv
python plots.py -coverage $SCRATCH/results/slcp-pamnre_cov.csv
