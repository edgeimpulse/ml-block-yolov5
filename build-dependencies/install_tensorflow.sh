#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    pip3 install tensorflow==2.7.0 -f https://tf.kmtea.eu/whl/stable.html
else
    pip3 install tensorflow==2.7.0
fi
