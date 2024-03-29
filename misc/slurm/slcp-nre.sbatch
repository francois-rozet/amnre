#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-nre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-nre_%a.log    # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amnre
cd ~/amnre

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -o $DST/models/slcp-nre.pth -device cuda

python eval.py $DST/models/slcp-nre.pth $SRC/slcp-test.h5 -o $DST/results/slcp-nre.csv
