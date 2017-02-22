#!/bin/bash
export THEANO_FLAGS=device=gpu0,floatX=float32
python ./train_nmt.py -m enfr



