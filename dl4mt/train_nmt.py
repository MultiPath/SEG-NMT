from nmt import train
from pprint import pprint

def setup():
    # home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        'saveto': home + '/.model/baseline_enfr.bs64.npz',
        'datasets': [home + '/.dataset/train.en.tok.shuf',
                     home + '/.dataset/train.fr.tok.shuf'],
        'valid_datasets': [home + '/.dataset/devset.en.tok',
                           home + '/.dataset/devset.fr.tok'],
        'dictionaries': [home + '/.dataset/train.en.tok.pkl',
                         home + '/.dataset/train.fr.tok.pkl'],

        'dim_word': 512,
        'dim': 1024,
        'n_words':     20000,
        'n_words_src': 20000,

        'optimizer': 'adam',
        'decay_c': 0.,
        'clip_c': 1.,
        'use_dropout': False,

        'lrate': 0.0001,

        'patience':1000,
        'maxlen': 50,
        'batch_size':64,
        'valid_batch_size':32,
        'validFreq':100,
        'dispFreq':10,
        'saveFreq':100,
        'sampleFreq':100,

        'overwrite': False,
        'reload_': True}
    return config


config = setup()
pprint(config)

validerr = train(**config)

print 'done'
