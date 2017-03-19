#!/bin/bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

# back-up the code
backup=./.backup/dl4mt-tm2_$timestamp
mkdir $backup
cp -r *.py $backup
cp -r *.sh $backup
echo 'backup code ok.'


export THEANO_FLAGS=device=gpu0,floatX=float32
python ./translate_gpu.py -m $1 -p round | tee $logfile
