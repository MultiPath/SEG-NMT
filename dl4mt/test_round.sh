#!/bin/bash
# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu$2,floatX=float32

python ./translate_gpu.py -m $1 -round | tee $logfile
