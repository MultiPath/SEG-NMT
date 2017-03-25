#!/usr/bin/env bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu,floatX=float32

python ./translate_multi.py -m $1 | tee $logfile
python ./score.py -m fren | tee -a $logfile
python ./split.py | tee -a $logfile
