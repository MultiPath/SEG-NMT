# setup the training and testing details in this file
def setup_fren():
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.A5'

    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok',          # source
                     home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok',          # target
                     home + '/.dataset/tm2.enfr/train.fr.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/tm2.enfr/train.en.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/tm2.enfr/devset.fr.tok',
                           home + '/.dataset/tm2.enfr/devset.en.tok',
                           home + '/.dataset/tm2.enfr/devset.fr.matched.tok',
                           home + '/.dataset/tm2.enfr/devset.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_fren.npz',

        # test phase
        'trans_from': home + '/.dataset/tm2.enfr/devset.fr.tok',
        'tm_source':  home + '/.dataset/tm2.enfr/devset.fr.matched.tok',
        'tm_target':  home + '/.dataset/tm2.enfr/devset.en.matched.tok',
        'trans_ref':  home + '/.dataset/tm2.enfr/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
    }
    return config


def setup_enfr():
    home  = '/home/thoma/work/TMNMT'
    model = '/home/thoma/scratch/tmnmt'
    #name  = 'TM2.v1'
    name  = 'TM2.A3'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/tm2.fren/train.en.top5.shuf.tok',          # source
                     home + '/.dataset/tm2.fren/train.fr.top5.shuf.tok',          # target
                     home + '/.dataset/tm2.fren/train.en.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/tm2.fren/train.fr.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/tm2.fren/devset.enfr.en.tok',
                           home + '/.dataset/tm2.fren/devset.enfr.fr.tok',
                           home + '/.dataset/tm2.fren/devset.enfr.en.matched.tok',
                           home + '/.dataset/tm2.fren/devset.enfr.fr.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/tm2.fren/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.fren/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.fren/train.fr.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_enfr.bs64.npz',

        # test phase
        'trans_from': home + '/.dataset/tm2.fren/devset.enfr.fr.tok',
        'tm_source':  home + '/.dataset/tm2.fren/devset.enfr.fr.matched.tok',
        'tm_target':  home + '/.dataset/tm2.fren/devset.enfr.en.matched.tok',
        'trans_ref':  home + '/.dataset/tm2.fren/devset.enfr.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
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
        'encoder':     'gru',
        'decoder':     'gru_cond',
        'dim_word':     512,
        'dim':          1024,

        # training details
        'optimizer':   'adam',
        'decay_c':      0.,
        'clip_c':       1.,
        'use_dropout':  False,
        'lrate':        0.0001,
        'patience':     1000,

        'batch_size':   32,
        'valid_batch_size': 32,

        'validFreq':    250,
        'bleuFreq':     5000,
        'saveFreq':     250,
        'sampleFreq':   20,

        'overwrite':    False,
        'reload_':      True,

        'use_pretrain': False,
        'only_train_g': False,
        'diagonal':     False,
        'eye':          False,
        'cos_sim':      False,
        'use_coverage': True,
        'nn_coverage':  False,
        'cov_dim':      10,

        'stochastic':   False,
        'build_gate':   True,
        'gate_loss':    False,
        'gate_lambda':  0.1,

        'disable_bleu': True,

        # testing details
        'beamsize':     5,
        'normalize':    True,
        'd_maxlen':     200,
        'check_bleu':   True
    }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())

    # get full model name
    config['saveto'] += '{}.{}.{}-{}.npz'.format(
            pair, 'ff' if config['use_pretrain'] else 'ss',
            config['batch_size'], config['maxlen']
        )

    return config
