from nmt import train
from pprint import pprint

def setup():
    home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    config = {
        'saveto': home + '/.model/baseline_fren.npz',
        'datasets': [home + '/.dataset/train.fr.tok.shuf',
                     home + '/.dataset/train.en.tok.shuf'],
        'valid_datasets': [home + '/.dataset/devset.fr.tok',
                           home + '/.dataset/devset.en.tok'],
        'dictionaries': [home + '/.dataset/train.fr.tok.pkl',
                         home + '/.dataset/train.en.tok.pkl'],

        'dim_word': 512,
        'dim': 1024,
        'n_words':     20000,
        'n_words_src': 20000,

        'optimizer': 'adam',
        'decay_c': 0.,
        'clip_c': 1.,
        'use_dropout': False,

        'lrate': 0.00002,

        'patience':1000,
        'maxlen': 50,
        'batch_size':32,
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
