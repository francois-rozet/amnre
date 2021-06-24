#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-pamnre        # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-pamnre.log      # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator HH -samples $SCRATCH/samples/hh-train.h5 -valid $SCRATCH/samples/hh-valid.h5 -model "$MODEL" -masks poisson -arbitrary -o $SCRATCH/models/hh-pamnre.pth -device cuda

python eval.py $SCRATCH/models/hh-pamnre.json $SCRATCH/samples/hh-event.h5 -masks =1 =2 =8 -o $SCRATCH/results/hh-pamnre-event.csv

python eval.py $SCRATCH/models/hh-pamnre.json $SCRATCH/samples/hh-test.h5 -masks =1 =2 -indices 0 64 -clean -consistency -o $SCRATCH/results/hh-pamnre_cons.csv
python eval.py $SCRATCH/models/hh-pamnre.json $SCRATCH/samples/hh-test.h5 -masks =1 -indices 0 16384 -clean -coverage -o $SCRATCH/results/hh-pamnre_cov.csv

python plots.py -loss $SCRATCH/models/hh-pamnre.csv
python plots.py -accuracy $SCRATCH/results/hh-pamnre_cons.csv -metrics probability
python plots.py -consistency $SCRATCH/results/hh-pamnre_cons_*.csv
python plots.py -coverage $SCRATCH/results/hh-pamnre_cov.csv
