#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=plots            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=plots.log          # Log-file
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G           # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=03:00:00             # Max execution time
#

conda activate amsi
cd $SCRATCH

mkdir -p plots

# Loss
files=$(find -path "**/models/*.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amsi/plots.py loss **/models/$file -o plots/${file/.csv/_loss.pdf}
done

# Errors
cp -r ~/amsi/misc/errors .

for file in errors/*.json; do
    python ~/amsi/plots.py error $file -o ${file/.json/.pdf}
done

# Consistency
files=$(find -path "**/results/*_cons.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amsi/plots.py consistency **/results/${file/.csv/}_*.csv -o plots/${file/.csv/.pdf}
done

# Coverage
files=$(find -path "**/results/*_cov.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amsi/plots.py coverage **/results/$file -o plots/${file/.csv/.pdf}
done

# ROC
files=$(find -path "**/results/*_roc.h5" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amsi/plots.py roc **/results/$file -o plots/${file/.h5/.pdf}
done

# Corners
cp -r ~/amsi/misc/corners .

for file in corners/*.json; do
    python ~/amsi/plots.py corner $file -o ${file/.json/.pdf}
done
