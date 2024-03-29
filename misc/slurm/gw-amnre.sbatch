#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=gw-amnre         # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=gw-amnre_%a.log    # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=24:00:00             # Max execution time
#
#SBATCH --array=1-3                 # Job array

conda activate amnre
cd ~/amnre

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"arch": "ResNet", "num_blocks": 17, "residual_size": 512, "activation": "ELU", "normalization": "batch"}'
FILTER='110111111001011'

python train.py -simulator GW -samples $SRC/gw-train.h5 -live -valid $SRC/gw-valid.h5 -model "$MODEL" -filter $FILTER -arbitrary -epochs 512 -descents 1024 -lr 2e-4 -scheduler cosine -o $DST/models/gw-amnre.pth -device cuda

python eval.py $DST/models/gw-amnre.pth $SRC/gw-event.h5 -masks =1 =2 -filter $FILTER -o $DST/results/gw-amnre-event.csv
python eval.py $DST/models/gw-amnre.pth $SRC/gw-test.h5 -masks =1 =2 -filter $FILTER -indices 0 64 -clean -consistency -o $DST/results/gw-amnre_cons.csv
python eval.py $DST/models/gw-amnre.pth $SRC/gw-test.h5 -masks =1 -filter $FILTER -indices 0 8192 -bins 256 -clean -calibration -o $DST/results/gw-amnre_cal.csv
