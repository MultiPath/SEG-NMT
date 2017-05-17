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

python ./translate_multi.py -m $1 -p round -mm $3 -start  25000 -end  75000 | tee $logfile-1 &
python ./translate_multi.py -m $1 -p round -mm $3 -start  75000 -end 125000 | tee $logfile-2 &
python ./translate_multi.py -m $1 -p round -mm $3 -start 125000 -end 175000 | tee $logfile-3 &
python ./translate_multi.py -m $1 -p round -mm $3 -start 175000 -end 225000 | tee $logfile-4 &
#python ./translate_multi.py -m $1 -p round -mm $3 -start 225000 -end 275000 | tee $logfile-5 &

wait
