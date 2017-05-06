#!/bin/bash

# Define a timestamp function
timestamp=$(date +"%Y-%m-%d_%T")
logfile=./.log/test_$timestamp.log # print timestamp
echo $logfile

export THEANO_FLAGS=device=gpu,floatX=float32

python ./translate_gpu.py -m fren_bleu -model '/root/disk/scratch/model-tmnmt/TM2.B7_fren.ss.32-50.npz' -step 130000 -p 'single' | tee $logfile
