#!/usr/bin/env bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu2,floatX=float32

python ./translate_multi.py -m $1 -mm $3 -i $2 | tee $logfile
python ./score.py -m $1 -mm $3 | tee -a $logfile
python ./split.py -m $1 -mm $3 | tee -a $logfile
