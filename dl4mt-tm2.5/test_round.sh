#!/bin/bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu$2,floatX=float32

# back-up the code
backup=./.backup/dl4mt-tm2_$timestamp
mkdir $backup
cp -r *.py $backup
cp -r *.sh $backup
echo 'backup code ok.'

python ./translate_multi.py -m $1 -p round -mm $3 -start 0 -end 25000 | tee $logfile
