#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=hh-data          # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=hh-data.log        # Log-file
#SBATCH --cpus-per-task=8           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=2G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=1-00:00:00           # Max execution time
#

cd ~/amsi/misc/

if [ -d "mecha" ]; then
    cd mecha/5_hh/
    conda activate mecha
else
    git clone https://github.com/mackelab/IdentifyMechanisticModels_2020 mecha
    cp hh_samples.py mecha/5_hh/
    cd mecha/5_hh/

    conda create -n mecha
    conda activate mecha
    conda install python
    pip install git+https://github.com/mackelab/delfi/
    pip install cython h5py

    cd model
    python compile.py build_ext --inplace
    cd ..
fi

python hh_samples.py -seed 0 -samples 1048576 -o $SCRATCH/samples/hh-train.h5
python hh_samples.py -seed 1 -samples 131072 -o $SCRATCH/samples/hh-test.h5
python hh_samples.py -seed 2 -samples 131072 -o $SCRATCH/samples/hh-valid.h5
python hh_samples.py -seed 3 -events -o $SCRATCH/samples/hh-event.h5
