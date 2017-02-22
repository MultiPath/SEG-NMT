# setup the training and testing details in this file

def setup_fren():
    home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train
        'saveto': home + '/.model/baseline_fren.bpe.npz',
        'datasets': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.shuf',
                     home + '/.dataset/fren.bpe/train.en.tok.bpe.shuf'],
        'valid_datasets': [home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.en.tok.bpe'],
        'dictionaries': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.en.tok.bpe.pkl'],

        # test
        'trans_from': home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
        'trans_ref':  home + '/.dataset/fren/devset.en.tok',
        'trans_to':   home + '/.translate/baseline_fren.bpe.valid'
        }

    return config

def setup_enfr():
    # home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train
        'saveto': home + '/.model/baseline_enfr.bs64.npz',
        'datasets': [home + '/.dataset/train.en.tok.shuf',
                     home + '/.dataset/train.fr.tok.shuf'],
        'valid_datasets': [home + '/.dataset/devset.en.tok',
                           home + '/.dataset/devset.fr.tok'],
        'dictionaries': [home + '/.dataset/train.en.tok.pkl',
                         home + '/.dataset/train.fr.tok.pkl'],

        # test
        'trans_from': home + '/.dataset/devset.en.tok',
        'trans_ref':  home + '/.dataset/devset.fr.tok',
        'trans_to':   home + '/translate/baseline_enfr.valid'
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
        'lrate': 0.00002,
        'patience':1000,

        'maxlen': 80,
        'batch_size':32,
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


