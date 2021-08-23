#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-npe         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-npe_%a.log    # Log-file
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

MODEL='{"num_transforms": 7, "hidden_features": 128, "num_blocks": 3}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -flow -o $DST/models/slcp-npe.pth -device cuda

python eval.py $DST/models/slcp-npe.pth $SRC/slcp-test.h5 -steps 1024 -burn 0 -o $DST/results/slcp-npe.csv
