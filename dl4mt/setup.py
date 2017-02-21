# setup the training and testing details in this file

def setup_fren():
    home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        'saveto': home + '/.model/baseline_fren.npz',
        'datasets': [home + '/.dataset/train.fr.tok.shuf',
                     home + '/.dataset/train.en.tok.shuf'],
        'valid_datasets': [home + '/.dataset/devset.fr.tok',
                           home + '/.dataset/devset.en.tok'],
        'dictionaries': [home + '/.dataset/train.fr.tok.pkl',
                         home + '/.dataset/train.en.tok.pkl'],

        'trans_to': home + '/.translate/baseline_fren.valid'
        }

    return config

def setup(pair='fren'):
    # basic setting
    config = {

        # model details
        'dim_word': 512,
        'dim': 1024,
        'n_words':     20000,
        'n_words_src': 20000,

        # training details
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
        'reload_': True,

        # testing details
        'beamsize': 5,
        'normalize': False
        }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())
    return config


