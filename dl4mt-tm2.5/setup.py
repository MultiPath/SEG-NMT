# setup files


def setup_fren_bpe():
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7.bpe'

    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/data_fren_new5_top2/train.fr.top5.shuf.tok.bpe',           # source
                     home + '/.dataset/data_fren_new5_top2/train.en.top5.shuf.tok.bpe',           # target
                     home + '/.dataset/data_fren_new5_top2/train.fr.top5.matched.shuf.tok.bpe1',  # source-TM1
                     home + '/.dataset/data_fren_new5_top2/train.en.top5.matched.shuf.tok.bpe1',  # target-TM1
                     home + '/.dataset/data_fren_new5_top2/train.fr.top5.matched.shuf.tok.bpe2',  # source-TM2
                     home + '/.dataset/data_fren_new5_top2/train.en.top5.matched.shuf.tok.bpe2'   # target-TM2
                     ],

        'valid_datasets': [home + '/.dataset/data_fren_new5_top2/devset.fr.tok.bpe',
                           home + '/.dataset/data_fren_new5_top2/devset.en.tok.bpe',
                           home + '/.dataset/data_fren_new5_top2/devset.fr.matched.tok.bpe1',
                           home + '/.dataset/data_fren_new5_top2/devset.en.matched.tok.bpe1',
                           home + '/.dataset/data_fren_new5_top2/devset.fr.matched.tok.bpe2',
                           home + '/.dataset/data_fren_new5_top2/devset.en.matched.tok.bpe2',
                           ],

        'dictionaries': [home + '/.dataset/data_fren_new5_top2/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/data_fren_new5_top2/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/data_fren_new5_top2/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/data_fren_new5_top2/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/data_fren_new5_top2/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/data_fren_new5_top2/train.en.top5.shuf.tok.bpe.pkl',
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,

        # baseline models
        'baseline_xy': model + '/baseline_fren.bpe.npz',

        # test phase
        'trans_from': home + '/.dataset/data_fren_new5_top2/devset.fr.tok.bpe',
        'tm_source':  home + '/.dataset/data_fren_new5_top2/devset.fr.matched.tok.bpe',
        'tm_target':  home + '/.dataset/data_fren_new5_top2/devset.en.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.fren/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/data_fren_new5_top2/train.fr.top1.tok.bpe',
        'tm_target_full': home + '/.dataset/data_fren_new5_top2/train.en.top1.tok.bpe',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'

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
        'see_pretrain': False,
        'only_train_g': False,
        'diagonal':     True,
        'eye':          True,
        'cos_sim':      False,
        'use_coverage': True,
        'nn_coverage':  False,
        'gate_coverage':True,
        'cov_dim':      10,
        'option':       'normal', # 'normal'

        'stochastic':   False,
        'build_gate':   True,
        'gate_loss':    False,
        'gate_lambda':  0.1,

        'disable_bleu': True,

        # testing details
        'beamsize':     5,
        'normalize':    True,
        'd_maxlen':     200,
        'check_bleu':   True,

        # remote monitor (tensorboard)
        'remote':       True,
        'address':      '147.8.182.14',
        'port':         8889
    }

    # get dataset info
    config.update(eval('setup_{}'.format(pair))())

    # get full model name
    config['saveto'] += '{}.{}.{}-{}.npz'.format(
            pair, 'ff' if config['use_pretrain'] else 'ss',
            config['batch_size'], config['maxlen']
        )
    print 'start {}'.format(config['saveto'])
    return config
