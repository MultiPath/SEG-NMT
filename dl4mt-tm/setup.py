# setup the training and testing details in this file
def setup_fren():
    home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'saveto': home + '/.model/baseline_fren.npz',
        'datasets': [home + '/.dataset/fren/train.fr.tok.shuf',  # source
                     home + '/.dataset/fren/train.en.tok.shuf',  # target
                     home + '/.dataset/fren/train.fr.tok.shuf',  # source-TM
                     home + '/.dataset/fren/train.en.tok.shuf'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/fren/devset.fr.tok',
                           home + '/.dataset/fren/devset.en.tok',
                           home + '/.dataset/fren/devset.fr.tok',
                           home + '/.dataset/fren/devset.en.tok'
                           ],

        'dictionaries': [home + '/.dataset/fren/train.fr.tok.pkl',
                         home + '/.dataset/fren/train.en.tok.pkl',
                         home + '/.dataset/fren/train.fr.tok.pkl',
                         home + '/.dataset/fren/train.en.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],

        # TODO: test phase is not ready
        # test phase
        'trans_from': home + '/.dataset/fren/devset.fr.tok',
        'trans_ref': home + '/.dataset/fren/devset.en.tok',
        'trans_to': home + '/.translate/baseline_fren.valid'
    }
    return config


def setup_fren_bpe():
    home = '/root/workspace/TMNMT'
    # home = '/home/thoma/work/TMNMT'
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'saveto': home + '/.model/tmv1_fren.bpe.npz',
        'datasets': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.shuf',  # source
                     home + '/.dataset/fren.bpe/train.en.tok.bpe.shuf',  # target
                     home + '/.dataset/fren.bpe/train.fr.tok.bpe.shuf',  # source-TM
                     home + '/.dataset/fren.bpe/train.en.tok.bpe.shuf'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.en.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/fren.bpe/devset.en.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/fren.bpe/train.fr.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.en.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.fr.tok.bpe.pkl',
                         home + '/.dataset/fren.bpe/train.en.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],

        # baseline models
        'baseline_fe': home + '/.model/baseline_fren.bpe.npz',
        'baseline_ef': home + '/.model/baseline_enfr.bpe.npz',

        # TODO: test phase is not ready
        # test phase
        'trans_from': home + '/.dataset/fren.bpe/devset.fr.tok.bpe',
        'trans_ref': home + '/.dataset/fren/devset.en.tok',
        'trans_to': home + '/.translate/tmv1_fren.bpe.valid'
    }
    return config


def setup(pair='fren'):
    # basic setting
    config = {

        # model details
        'encoder': 'gru',
        'decoder': 'gru_cond',
        'dim_word': 512,
        'dim': 1024,

        # training details
        'optimizer': 'adam',
        'decay_c': 0.,
        'clip_c': 1.,
        'use_dropout': False,
        'lrate': 0.00002,
        'patience': 1000,

        'maxlen': 80,
        'batch_size': 32,
        'valid_batch_size': 32,
        'validFreq': 100,
        'dispFreq': 10,
        'saveFreq': 100,
        'sampleFreq': 100,

        'overwrite': False,
        'reload_': True,

        'use_pretrain': False,
        'stochastic': True,

        # testing details
        'beamsize': 5,
        'normalize': False,
        'd_maxlen': 200
    }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())
    return config
