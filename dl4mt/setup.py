# setup the training and testing details in this file

def setup_fren_bpe():
    home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    # home   = '/scratch/jg5223/exp/TMNMT'
    # home  = '/root/workspace/TMNMT'
    # model = '/root/disk/scratch/model-tmnmt/'
    config = {
        # train
        'saveto': model + '/baseline_fren.bpe.npz',
        'datasets': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.shuf',
                     home + '/.dataset/fren.bpe/train.en.tok.bpe.shuf'],
        'valid_datasets': [home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.en.tok.bpe'],
        'dictionaries': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.en.tok.bpe.pkl'],

        # test
        'trans_from': home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
        'trans_ref':  home + '/.dataset/fren/devset.en.tok',
        'trans_to':   home + '/.translate/baseline_fren.bpe.valid2'
        }

    return config


def setup_enfr_bpe():
    home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train
        'saveto': home + '/.model/baseline_enfr.bpe1.npz',
        'datasets': [home + '/.dataset/fren.bpe/train.en.tok.bpe.shuf',
                     home + '/.dataset/fren.bpe/train.fr.tok.bpe.shuf'],
        'valid_datasets': [home + '/.dataset/fren.bpe/devset.en.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.fr.tok.bpe'],
        'dictionaries': [home + '/.dataset/fren.bpe/train.en.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.fr.tok.bpe.pkl'],

        # test
        'trans_from': home + '/.dataset/fren.bpe/devset.en.tok.bpe',
        'trans_ref':  home + '/.dataset/fren/devset.fr.tok',
        'trans_to':   home + '/.translate/baseline_enfr.bpe.valid'
        }

    return config


def setup_fren():
    # home   = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt/'
    config = {
        # train
        'saveto': model + 'baseline_fren.npz',
        'datasets': [home + '/.dataset/fren/train.fr.tok.shuf',
                     home + '/.dataset/fren/train.en.tok.shuf'],
        'valid_datasets': [home + '/.dataset/fren/devset.fr.tok',
                           home + '/.dataset/fren/devset.en.tok'],
        'dictionaries': [home + '/.dataset/fren/train.fr.tok.pkl',
                         home + '/.dataset/fren/train.en.tok.pkl'],

        # test
        'trans_from': home + '/.dataset/fren/devset.fr.tok',
        'trans_ref':  home + '/.dataset/fren/devset.en.tok',
        'trans_to':   home + '/translate/baseline_fren.valid2'
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

        'maxlen': 80,
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
        'normalize': False,
        'd_maxlen': 200,

        # remote monitor (tensorboard)
        'remote':       True,
        'address':      '147.8.182.14',
        'port':         8889

        }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())
    return config


