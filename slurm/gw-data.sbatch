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

python samples.py -simulator GW -seed 0 -events -o $SCRATCH/samples/gw-event.h5
python samples.py -simulator GW -seed 1 -samples 1048576 -basis-from $SCRATCH/samples/gw-event.h5 -o $SCRATCH/samples/gw-train.h5
python samples.py -simulator GW -seed 2 -samples 131072 -basis-from $SCRATCH/samples/gw-event.h5 -o $SCRATCH/samples/gw-test.h5
python samples.py -simulator GW -seed 3 -samples 131072 -basis-from $SCRATCH/samples/gw-event.h5 -o $SCRATCH/samples/gw-valid.h5
