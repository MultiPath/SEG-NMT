#!/bin/bash
export THEANO_FLAGS=device=gpu1,floatX=float32

python ./translate_gpu.py -m fren
python ./score.py -m fren

