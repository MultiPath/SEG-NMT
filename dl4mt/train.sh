#!/bin/bash
export THEANO_FLAGS=device=gpu1,floatX=float32
python ./train_nmt.py -m enfr_bpe



