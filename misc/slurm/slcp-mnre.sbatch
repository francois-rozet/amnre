#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=slcp-mnre        # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=slcp-mnre_%a.log   # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=06:00:00             # Max execution time
#
#SBATCH --array=1-5                 # Job array

conda activate amsi
cd ~/amsi

SRC=$SCRATCH/samples
DST=$SCRATCH/$SLURM_ARRAY_TASK_ID

MODEL='{"num_layers": 5, "hidden_size": 128, "activation": "ELU"}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -masks =1 =2 -o $DST/models/slcp-mnre.pth -device cuda

python eval.py $DST/models/slcp-mnre.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 3 -o $DST/results/slcp-mnre.csv
python eval.py $DST/models/slcp-mnre.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $DST/results/slcp-mnre_cons.csv
python eval.py $DST/models/slcp-mnre.pth $SRC/slcp-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -coverage -o $DST/results/slcp-mnre_cov.csv
