#!/bin/bash
# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu0,floatX=float32

python ./translate_gpu.py -m fren_bpe -round True | tee $logfile
