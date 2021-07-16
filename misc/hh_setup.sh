#!/usr/bin/env bash
#
# conda activate amnre

CACHE=~/.cache/IdentifyMechanisticModels

if ! [ -d $CACHE ]; then
    git clone https://github.com/mackelab/IdentifyMechanisticModels_2020 $CACHE
fi

rm -r amnre/simulators/hhpkg
cp -r $CACHE/5_hh/model amnre/simulators/hhpkg

cd amnre/simulators/hhpkg

for file in *.py; do
    sed -i 's/import model\./from . import /g' $file
    sed -i 's/from model\./from ./g' $file
done

pip install cython
pip install git+https://github.com/mackelab/delfi

python compile.py build_ext --inplace
