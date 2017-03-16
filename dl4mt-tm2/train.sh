#!/bin/bash


# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/train_$timestamp.log # print timestamp
echo $logfile


export THEANO_FLAGS=device=gpu0,floatX=float32
python ./train_nmt.py -m $1 | tee $logfile
# python ./train_nmt2.py -m fren | tee $logfile

