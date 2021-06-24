#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-data          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-data.log        # Log-file
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=2G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=1-00:00:00           # Max execution time
#

conda activate amsi

cd ~/amsi

python sample.py -simulator HH -seed 0 -events -o $SCRATCH/samples/hh-event.h5
python sample.py -simulator HH -seed 1 -samples 1048576 -reference $SCRATCH/samples/hh-event.h5 -o $SCRATCH/samples/hh-train.h5
python sample.py -simulator HH -seed 2 -samples 131072 -reference $SCRATCH/samples/hh-event.h5 -o $SCRATCH/samples/hh-test.h5
python sample.py -simulator HH -seed 3 -samples 131072 -reference $SCRATCH/samples/hh-event.h5 -o $SCRATCH/samples/hh-valid.h5
