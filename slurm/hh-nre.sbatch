#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-nre           # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-nre.log         # Log-file
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#

conda activate amsi

cd ~/amsi

MODEL='{"num_layers": 10, "hidden_size": 256, "activation": "SELU"}'

python train.py -simulator HH -samples $SCRATCH/samples/hh-train.h5 -valid $SCRATCH/samples/hh-valid.h5 -model "$MODEL" -o $SCRATCH/models/hh-nre.pth -device cuda

python eval.py $SCRATCH/models/hh-nre.json $SCRATCH/samples/hh-event.h5 -plots -o $SCRATCH/results/hh-nre-event.csv
python eval.py $SCRATCH/models/hh-nre.json $SCRATCH/samples/hh-test.h5 -indices 0 3 -plots -o $SCRATCH/results/hh-nre.csv
