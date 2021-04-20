#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-data          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-data.log        # Log-file
#SBATCH --cpus-per-task=8           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=2-00:00:00           # Max execution time
#

conda activate amsi

cd /home/frozet/amsi/python
python samples.py -simulator GW -seed 1 -samples 4194304 -basis-from $SCRATCH/samples/gw-basis.h5 -o $SCRATCH/samples/gw-train.h5
python samples.py -simulator GW -seed 2 -samples 131072 -basis-from $SCRATCH/samples/gw-basis.h5 -o $SCRATCH/samples/gw-test.h5
