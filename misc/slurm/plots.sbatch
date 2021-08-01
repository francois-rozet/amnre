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

conda activate amnre
cd $SCRATCH

mkdir -p plots

# Loss
files=$(find -path "**/models/*.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amnre/plots.py loss **/models/$file -o plots/${file/.csv/_loss.pdf}
done

# Errors
cp -r ~/amnre/misc/errors .

for file in errors/*.json; do
    python ~/amnre/plots.py error $file -o ${file/.json/.pdf}
done

# Consistency
files=$(find -path "**/results/*_cons.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amnre/plots.py consistency **/results/${file/.csv/}_*.csv -o plots/${file/.csv/.pdf}
done

# Coverage
files=$(find -path "**/results/*_cov.csv" | xargs basename -a | sort -u)

for file in $files; do
    python ~/amnre/plots.py coverage **/results/$file -o plots/${file/.csv/.pdf}
done

# ROC
cp -r ~/amnre/misc/rocs .

for file in rocs/*.json; do
    python ~/amnre/plots.py roc $file -o ${file/.json/.pdf}
done

# Corners
cp -r ~/amnre/misc/corners .

for file in corners/*.json; do
    python ~/amnre/plots.py corner $file -o ${file/.json/.pdf}
done
