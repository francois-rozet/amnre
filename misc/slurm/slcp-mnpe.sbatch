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

MODEL='{"design": "MAF", "num_transforms": 5, "hidden_features": 64, "num_blocks": 2}'
EMBED='{"name": "MLP", "num_layers": 5, "hidden_size": 128, "output_size": 128, "activation": "ELU"}'

python train.py -simulator SLCP -samples $SRC/slcp-train.h5 -valid $SRC/slcp-valid.h5 -model "$MODEL" -embedding "$EMBED" -flow -masks =1 =2 -o $DST/models/slcp-mnpe.pth -device cuda

python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 3 -o $DST/results/slcp-mnpe.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 =2 -indices 0 64 -clean -accuracy -consistency -o $DST/results/slcp-mnpe_cons.csv
python eval.py $DST/models/slcp-mnpe.pth $SRC/slcp-test.h5 -masks =1 -indices 0 8192 -bins 256 -clean -coverage -o $DST/results/slcp-mnpe_cov.csv
