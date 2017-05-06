#!/bin/bash
export THEANO_FLAGS=device=gpu$2,floatX=float32

python ./translate_gpu.py -m $1
python ./score.py -m $1


