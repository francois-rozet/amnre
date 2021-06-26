#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-mnre          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-mnre.log        # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=12:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

MODEL='{"num_layers": 3, "hidden_size": 64, "activation": "SELU"}'
ENCODER='{"name": "MLP", "output_size": 256, "num_layers": 7, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator GW -samples $SCRATCH/samples/gw-train.h5 -valid $SCRATCH/samples/gw-valid.h5 -model "$MODEL" -encoder "$ENCODER" -masks =1 =2 -o $SCRATCH/models/gw-mnre.pth -device cuda

python eval.py $SCRATCH/models/gw-mnre.json $SCRATCH/samples/gw-event.h5 -masks =1 =2 -o $SCRATCH/results/gw-mnre-event.csv

python eval.py $SCRATCH/models/gw-mnre.json $SCRATCH/samples/gw-test.h5 -masks =1 =2 -indices 0 64 -clean -consistency -o $SCRATCH/results/gw-mnre_cons.csv
python eval.py $SCRATCH/models/gw-mnre.json $SCRATCH/samples/gw-test.h5 -masks =1 -indices 0 16384 -clean -coverage -o $SCRATCH/results/gw-mnre_cov.csv

python plots.py -loss $SCRATCH/models/gw-mnre.csv
python plots.py -accuracy $SCRATCH/results/gw-mnre_cons.csv -metrics probability
python plots.py -consistency $SCRATCH/results/gw-mnre_cons_*.csv
python plots.py -coverage $SCRATCH/results/gw-mnre_cov.csv
