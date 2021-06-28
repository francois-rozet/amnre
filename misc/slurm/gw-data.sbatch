#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-data          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-data.log        # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=1-00:00:00           # Max execution time
#

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples

python sample.py -simulator GW -seed 0 -chunk-size 32768 -events -o $SRC/gw-event.h5
python sample.py -simulator GW -seed 1 -reference $SRC/gw-event.h5 -samples 4194304 -o $SRC/gw-train.h5
python sample.py -simulator GW -seed 2 -reference $SRC/gw-event.h5 -samples 131072 -o $SRC/gw-test.h5
python sample.py -simulator GW -seed 3 -reference $SRC/gw-event.h5 -samples 131072 -o $SRC/gw-valid.h5
