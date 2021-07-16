#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-bilby         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-bilby.log       # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=3-00:00:00           # Max execution time
#

conda activate amnre
cd $SCRATCH

cp -r ~/amnre/misc/references .
cd references

python bilby_train.py
python bilby_eval.py
