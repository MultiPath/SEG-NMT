#!/usr/bin/env bash

model=".pretrained/model_wmt15_bpe2k_basic_cs-en.npz"
dict="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/csen/train/all_cs-en.cs.tok.bpe.word.pkl"
dict_rev="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/csen/train/all_cs-en.en.tok.bpe.word.pkl"
source="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/csen/dev/newstest2013-ref.cs.tok.bpe"
reference="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/csen/dev/newstest2013-src.en.tok"

for beamsize in 2 3 7 15 20
do
    echo 'use beamsize='$beamsize
    saveto=".translate/standard.trans.cs_en.b=$beamsize"
    THEANO_FLAGS="floatX=float32, device=cpu" python translate.py -k $beamsize -p 49 $model $dict $dict_rev $source $saveto
    sed -i 's/@@ //g' $saveto
    ./data/multi-bleu.perl $reference < $saveto | tee ".translate/score-beam=$beamsize" 
done

