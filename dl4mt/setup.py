# setup the training and testing details in this file
def setup_fren_bpe_28():
    home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'baseline.bpe.'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.28.fren.bpe/train.fr.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/top5k.28.fren.bpe/train.en.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/top5k.28.fren.bpe/train.fr.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/top5k.28.fren.bpe/train.en.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.28.fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/top5k.28.fren.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.28.fren.bpe/devset.fr.matched.tok.bpe',
                           home + '/.dataset/top5k.28.fren.bpe/devset.en.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/top5k.28.fren.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.28.fren.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.28.fren.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.28.fren.bpe/train.en.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,

        'trans_from': home + '/.dataset/top5k.28.fren.bpe/devset.fr.tok.bpe',
        'tm_source':  home + '/.dataset/top5k.28.fren.bpe/devset.fr.matched.tok.bpe',
        'tm_target':  home + '/.dataset/top5k.28.fren.bpe/devset.en.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.fren.bpe/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren.bpe/train.fr.top1.tok.bpe',
        'tm_target_full': home + '/.dataset/top5k.fren.bpe/train.en.top1.tok.bpe',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'

    }
    return config


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


def setup_esen_bpe_latest():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'baseline.latest4.esen'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '.npz',
        'datasets': [home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe',          # target
                     ],

        'valid_datasets': [home + '/.dataset/latest4.esen.bpe/devset.es.tok.bpe',
                           home + '/.dataset/latest4.esen.bpe/devset.en.tok.bpe',
                           ],

        'dictionaries': [home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         ],

        'maxlen': 80,
        'address': '147.8.182.14',

        # test phase
        'trans_from': home + '/.dataset/latest4.esen.bpe/devset.es.tok.bpe',
        'trans_ref':  home + '/.dataset/latest4.esen.bpe/dev.esen.en.tok',
        'trans_to':   home + '/.translate/' + name + '.esen_bpe.dev.translate',

    }
    return config


def setup_enes_bpe_latest():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'baseline.latest4'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '.npz',
        'datasets': [home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe',          # target
                     ],

        'valid_datasets': [home + '/.dataset/latest4.esen.bpe/devset.en.tok.bpe',
                           home + '/.dataset/latest4.esen.bpe/devset.es.tok.bpe',
                           ],

        'dictionaries': [home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         ],

        'maxlen': 80,
        'address': '147.8.182.14',

        # test phase
        'trans_from': home + '/.dataset/latest4.esen.bpe/devset.en.tok.bpe',
        'trans_ref':  home + '/.dataset/latest4.esen.bpe/dev.esen.es.tok',
        'trans_to':   home + '/.translate/' + name + '.enes_bpe.dev.translate',

    }
    return config




def setup_enes_bpe_miles():
    home  = '/home/thoma/work/TMNMT'
    model = '/home/thoma/scratch/tmnmt'
    name  = 'baseline.miles'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.enes.bpe/train.en.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/top5k.enes.bpe/train.es.top5.shuf.tok.bpe',          # target
                     ],

        'valid_datasets': [home + '/.dataset/top5k.enes.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.enes.bpe/devset.es.tok.bpe',
                           ],

        'dictionaries': [home + '/.dataset/top5k.enes.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enes.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         ],

        'maxlen': 80,

        # test phase
        'trans_from': home + '/.dataset/top5k.enes.bpe/devset.en.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.enes.bpe/devset.es.tok',
        'trans_to':   home + '/.translate/' + name + '.enes_bpe.dev.translate',

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
        'validFreq':1000,
        'dispFreq':10,
        'saveFreq':1000,
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


