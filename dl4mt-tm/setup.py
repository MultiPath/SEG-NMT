# setup the training and testing details in this file
def setup_fren():
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt' 
    
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'saveto': model + '/new.M-gate.v1_',
        'datasets': [home + '/.dataset/tm.fren/train.fr.top5.random5.shuf.tok',          # source
                     home + '/.dataset/tm.fren/train.en.top5.random5.shuf.tok',          # target
                     home + '/.dataset/tm.fren/train.fr.top5.random5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/tm.fren/train.en.top5.random5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/tm.fren/devset.fr.tok',
                           home + '/.dataset/tm.fren/devset.en.tok',
                           home + '/.dataset/tm.fren/devset.fr.matched.tok',
                           home + '/.dataset/tm.fren/devset.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/tm.fren/train.fr.top5.random5.shuf.tok.pkl',
                         home + '/.dataset/tm.fren/train.en.top5.random5.shuf.tok.pkl',
                         home + '/.dataset/tm.fren/train.fr.top5.random5.shuf.tok.pkl',
                         home + '/.dataset/tm.fren/train.en.top5.random5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_fren.npz',
        'baseline_yx': model + '/baseline_enfr.bs64.npz',
        
        # TODO: test phase is not ready
        # test phase
        'trans_from': home + '/.dataset/fren/devset.fr.tok',
        'trans_ref':  home + '/.dataset/fren/devset.en.tok',
        'trans_to':   home + '/.translate/tmv3_'
    }
    return config


def setup_fren_bpe():
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt' 
    # home = '/home/thoma/work/TMNMT'
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'saveto': model + '/tmv2_',
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
        'maxlen': 80,

        # baseline models
        'baseline_ef': model + '/baseline_fren.bpe.npz',
        'baseline_fe': model + '/baseline_enfr.bpe.npz',

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
        'lrate': 0.000002,
        'patience': 1000,

        'batch_size': 16,
        'valid_batch_size': 32,
        'validFreq': 100,
        'dispFreq': 10,
        'saveFreq': 500,
        'sampleFreq': 20,

        'overwrite': True,
        'reload_': True,

        'use_pretrain': True,
        'stochastic': True,
        'build_gate': True,
        'gate_loss': False,
        'gate_lambda': 0.01,

        # testing details
        'beamsize': 5,
        'normalize': False,
        'd_maxlen': 200
    }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())
    
    # get full model name
    config['saveto'] += '{}.{}.{}-{}.npz'.format(
            pair, 'ff' if config['use_pretrain'] else 'ss',
            config['batch_size'], config['maxlen']
        )
    
    return config
