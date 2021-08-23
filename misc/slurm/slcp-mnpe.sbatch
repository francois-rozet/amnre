#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-mnpe        # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-mnpe_%a.log   # Log-file
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

MODEL='{"num_transforms": 7, "hidden_features": 128, "num_blocks": 3}'
ADV='{"num_layers": 11, "hidden_size": 512, "activation": "ELU"}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -flow -masks =1 =2 =5 -o $DST/models/slcp-mnpe.pth -device cuda
python train.py -simulator SLCP -samples $SRC/slcp-adv.h5 -valid $SRC/slcp-valid.h5 -model "$ADV" -masks 00100 00110 11111 -adversary $DST/models/slcp-mnpe.pth -lr 1e-4 -min-lr 1e-5 -patience 11 -o $DST/models/slcp-mnpe-adv.pth -device cuda

python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 =2 =5 -steps 1024 -burn 0 -o $DST/results/slcp-mnpe.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $DST/results/slcp-mnpe_cons.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -calibration -o $DST/results/slcp-mnpe_cal.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks 00100 00110 11111 -indices 0 0 -kl -o $DST/results/slcp-mnpe_kl.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks 00100 00110 11111 -indices 0 0 -classify -o $DST/results/slcp-mnpe_roc.csv
python eval.py $DST/models/slcp-mnpe-adv.pth $SRC/slcp-test.h5 -masks 00100 00110 11111 -indices 0 0 -classify -o $DST/results/slcp-mnpe-adv_roc.csv
