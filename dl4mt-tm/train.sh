#!/bin/bash


# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/$timestamp.log # print timestamp
echo $logfile


export THEANO_FLAGS=device=gpu0,floatX=float32
python ./train_nmt.py -m fren | tee $logfile

