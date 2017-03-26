#!/bin/bash


# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/train_$timestamp.log # print timestamp
echo $logfile

# back-up the code
backup=./.backup/dl4mt-tm2_$timestamp
mkdir $backup
cp -r *.py $backup
cp -r *.sh $backup
echo 'backup code ok.'

export THEANO_FLAGS=device=gpu1,floatX=float32
python ./train_nmt.py -m $1 | tee $logfile
# python ./train_nmt2.py -m fren | tee $logfile

