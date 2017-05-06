#!/usr/bin/env bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu,floatX=float32

python ./translate_multi.py -m $1 -mm 1 -i $2 -ss | tee $logfile
python ./score.py -m $1 -mm 1 -ss | tee -a $logfile
python ./split.py -m $1 -mm 1 -ss | tee -a $logfile
