#!/bin/sh
set -e

KUNGFU_MINDSPORE=$HOME/Mindspore/kungfu-mindspore

. $KUNGFU_MINDSPORE/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $KUNGFU_MINDSPORE/mindspore)

mccmp() {
    echo "file1: $1"
    echo "file2: $2"
    python3.7 ./mindspore-compare-checkpoints.py $1 $2
}

mccmp \
    ./checkpoint_0/final-train-net.ckpt \
    ./checkpoint_1/final-train-net.ckpt \
    > diff-init-final-net.txt
