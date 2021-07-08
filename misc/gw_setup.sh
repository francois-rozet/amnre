#!/usr/bin/env bash
#
# conda activate amsi

pip install gwpy pycbc bilby

CACHE=~/.cache/lfi-gw

if ! [ -d $CACHE ]; then
    git clone https://github.com/stephengreen/lfi-gw $CACHE
    jupyter nbconvert --to notebook --execute $CACHE/notebooks/GW150914_data.ipynb --inplace --allow-errors
fi

cp -r $CACHE/lfigw amsi/simulators
cp -r $CACHE/data/* amsi/simulators/lfigw
cp $CACHE/bilby_runs/bilby_GW150914.py misc/references/bilby_train.py
