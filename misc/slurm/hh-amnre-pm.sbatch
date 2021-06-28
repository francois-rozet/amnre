#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-amnre-pm      # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-amnre-pm_%a.log # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'

python train.py -simulator HH -samples $SRC/hh-train.h5 -valid $SRC/hh-valid.h5 -model "$MODEL" -arbitrary -masks poisson -o $DST/models/hh-amnre-pm.pth -device cuda

python eval.py $DST/models/hh-amnre-pm.pth $SRC/hh-event.h5 -masks =1 =2 =8 -o $DST/results/hh-amnre-pm-event.csv
python eval.py $DST/models/hh-amnre-pm.pth $SRC/hh-test.h5 -masks =1 =2 -indices 0 64 -clean -consistency -o $DST/results/hh-amnre-pm_cons.csv
python eval.py $DST/models/hh-amnre-pm.pth $SRC/hh-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -coverage -o $DST/results/hh-amnre-pm_cov.csv
