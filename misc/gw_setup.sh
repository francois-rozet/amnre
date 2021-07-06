#!/usr/bin/env bash
#
# conda activate amsi

pip install gwpy pycbc

CACHE=~/.cache/lfi-gw

if ! [ -d $CACHE ]; then
    git clone https://github.com/stephengreen/lfi-gw $CACHE
    jupyter nbconvert --to notebook --execute $CACHE/notebooks/GW150914_data.ipynb --inplace --allow-errors
fi

rm -r amsi/simulators/lfigw
cp -r $CACHE/lfigw amsi/simulators/lfigw
cp -r $CACHE/data/* amsi/simulators/lfigw
