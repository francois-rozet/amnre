#!/usr/bin/env bash
#
# conda activate amnre

pip install gwpy pycbc bilby

CACHE=~/.cache/lfi-gw

if ! [ -d $CACHE ]; then
    git clone https://github.com/francois-rozet/lfi-gw $CACHE
fi

cp -r $CACHE/lfigw amnre/simulators
cp -r $CACHE/data/* amnre/simulators/lfigw
cp -r $CACHE/bilby_runs/* misc/references
