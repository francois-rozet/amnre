#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-mnre          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-mnre_%a.log     # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=12:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amnre
cd ~/amnre

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 7, "hidden_size": 256, "activation": "ELU"}'
ADV='{"num_layers": 11, "hidden_size": 512, "activation": "ELU"}'

python train.py -simulator HH -samples $SRC/hh-train.h5 -valid $SRC/hh-valid.h5 -model "$MODEL" -masks =1 =2 11100100 11111111 -o $DST/models/hh-mnre.pth -device cuda
python train.py -simulator HH -samples $SRC/hh-adv.h5 -valid $SRC/hh-valid.h5 -model "$ADV" -masks 10000000 11000000 11100100 11111111 -adversary $DST/models/hh-mnre.pth -lr 1e-4 -min-lr 1e-5 -patience 11 -o $DST/models/hh-mnre-adv.pth -device cuda

python eval.py $DST/models/hh-mnre.pth $SRC/hh-event.h5 -masks =1 =2 -o $DST/results/hh-mnre-event.csv
python eval.py $DST/models/hh-mnre.pth $SRC/hh-test.h5 -masks =1 =2 -indices 0 64 -clean -consistency -o $DST/results/hh-mnre_cons.csv
python eval.py $DST/models/hh-mnre.pth $SRC/hh-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -calibration -o $DST/results/hh-mnre_cal.csv
python eval.py $DST/models/hh-mnre.pth $SRC/hh-test.h5 -masks 10000000 11000000 11100100 11111111 -indices 0 0 -classify -o $DST/results/hh-mnre_roc.csv
python eval.py $DST/models/hh-mnre-adv.pth $SRC/hh-test.h5 -masks 10000000 11000000 11100100 11111111 -indices 0 0 -classify -o $DST/results/hh-mnre-adv_roc.csv
