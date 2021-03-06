#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/dz/raintest
DATA=/home/dz/raintest
TOOLS=/home/dz/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/raintest_train_lmdb \
  $DATA/raintest_mean.binaryproto

echo "Done."
