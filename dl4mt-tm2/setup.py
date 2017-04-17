# setup the training and testing details in this file
def setup_fren0():
    home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    #home  = '/root/workspace/TMNMT'
    #model = '/root/disk/scratch/model-tmnmt'
    # name  = 'TM2.A0'
    name  = 'TM2.B5'

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

        'valid_datasets': [home + '/.dataset/tm2.fren/devset.fr.tok',
                           home + '/.dataset/tm2.fren/devset.en.tok',
                           home + '/.dataset/tm2.fren/devset.fr.matched.tok',
                           home + '/.dataset/tm2.fren/devset.en.matched.tok'
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
        'trans_from': home + '/.dataset/tm2.fren/devset.fr.tok',
        'tm_source':  home + '/.dataset/tm2.fren/devset.fr.matched.tok',
        'tm_target':  home + '/.dataset/tm2.fren/devset.en.matched.tok',
        'trans_ref':  home + '/.dataset/tm2.fren/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
    }
    return config


def setup_ende_wmt15():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.copy_scratch'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/wmt15.ende/train_copy/train.en.top5.copy.tok.bpe',          # source
                     home + '/.dataset/wmt15.ende/train_copy/train.de.top5.copy.tok.bpe',          # target
                     home + '/.dataset/wmt15.ende/train_copy/train.en.top5.copy.matched.tok.bpe',  # source-TM
                     home + '/.dataset/wmt15.ende/train_copy/train.de.top5.copy.matched.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,
        'batch_size': 64,
        'lrate':      0.0001, # try this very small learning rate.

        # special care
        'dim': 1028,
        'use_pretrain': False,
        'see_pretrain': True,

        # baseline models
        'baseline_xy': model + '/model_wmt15_bi_en-de.npz',

        # test phase
        'trans_from': home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok.bpe',
        'tm_source':  home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok',
        'trans_to':   home + '/.translate/' + name + '.wmt15.dev.translate'
    }
    return config


def setup_ende_wmt():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.new_scratch'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/wmt15.ende/train/train.en.top5.tok.bpe',          # source
                     home + '/.dataset/wmt15.ende/train/train.de.top5.tok.bpe',          # target
                     home + '/.dataset/wmt15.ende/train/train.en.top5.matched.tok.bpe',  # source-TM
                     home + '/.dataset/wmt15.ende/train/train.de.top5.matched.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,
        'batch_size': 64,
        'lrate':      0.0001, # try this very small learning rate.

        # special care
        'dim': 1028,
        'use_pretrain': False,
        'see_pretrain': True,

        # baseline models
        'baseline_xy': model + '/model_wmt15_bi_en-de.npz',

        # test phase
        'trans_from': home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok.bpe',
        'tm_source':  home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok',
        'trans_to':   home + '/.translate/' + name + '.wmt15.dev.translate'
    }
    return config


def setup_deen_wmt():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.new_scratch'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/wmt15.ende/train/train.de.top5.tok.bpe',          # source
                     home + '/.dataset/wmt15.ende/train/train.en.top5.tok.bpe',          # target
                     home + '/.dataset/wmt15.ende/train/train.de.top5.matched.tok.bpe',  # source-TM
                     home + '/.dataset/wmt15.ende/train/train.en.top5.matched.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe',
                           home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.de.tok.bpe.word.pkl',
                         home + '/.dataset/wmt15.ende/train/all_de-en.en.tok.bpe.word.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,
        'batch_size': 64,
        'lrate':      0.0001, # try this very small learning rate.

        # special care
        'dim': 1028,
        'use_pretrain': False,
        'see_pretrain': False,

        # baseline models
        'baseline_xy': model + '/model_wmt15_bi_de-en.npz',

        # test phase
        'trans_from': home + '/.dataset/wmt15.ende/dev/newstest2013.de.tok.bpe',
        'tm_source':  home + '/.dataset/wmt15.ende/dev/devset.de.matched.tok.bpe',
        'tm_target':  home + '/.dataset/wmt15.ende/dev/devset.en.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/wmt15.ende/dev/newstest2013.en.tok',
        'trans_to':   home + '/.translate/' + name + 'deen.wmt15.dev.translate'
    }
    return config




def setup_deen():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    # home  = '/root/workspace/TMNMT'
    # model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.deen/train.de.top5.shuf.tok',          # source
                     home + '/.dataset/top5k.deen/train.en.top5.shuf.tok',          # target
                     home + '/.dataset/top5k.deen/train.de.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/top5k.deen/train.en.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.deen/dev.deen.de.tok',
                           home + '/.dataset/top5k.deen/dev.deen.en.tok',
                           home + '/.dataset/top5k.deen/devset.de.matched.tok',
                           home + '/.dataset/top5k.deen/devset.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/top5k.deen/train.de.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.deen/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.deen/train.de.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.deen/train.en.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_deen.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.deen/dev.deen.de.tok',
        'tm_source':  home + '/.dataset/top5k.deen/devset.de.matched.tok',
        'tm_target':  home + '/.dataset/top5k.deen/devset.en.matched.tok',
        'trans_ref':  home + '/.dataset/top5k.deen/dev.deen.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
    }
    return config


def setup_enes_bpe_miles():
    home  = '/home/thoma/work/TMNMT'
    model = '/home/thoma/scratch/tmnmt'
    name  = 'TM2.B7.miles3'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.enes.bpe/train.en.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/top5k.enes.bpe/train.es.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/top5k.enes.bpe/train.en.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/top5k.enes.bpe/train.es.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.enes.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.enes.bpe/devset.es.tok.bpe',
                           home + '/.dataset/top5k.enes.bpe/devset.en.matched.tok.bpe',
                           home + '/.dataset/top5k.enes.bpe/devset.es.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/top5k.enes.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enes.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enes.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enes.bpe/train.es.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,
        'skip': 5000 / 32,

        # baseline models
        'baseline_xy': model + '/baseline_enes.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.enes.bpe/devset.en.tok.bpe',
        'tm_source':  home + '/.dataset/top5k.enes.bpe/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/top5k.enes.bpe/devset.es.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.enes.bpe/devset.es.tok',
        'trans_to':   home + '/.translate/' + name + '.enes_bpe.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.enes/train.en.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.enes/train.es.top1.tok',
        'tm_rank': home + '/.dataset/top5k.enes/match_top100.pkl'
    }
    return config


def setup_esen_bpe_latest():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.latest4'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/latest4.esen.bpe/train.es.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/latest4.esen.bpe/train.en.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/latest4.esen.bpe/devset.es.tok.bpe',
                           home + '/.dataset/latest4.esen.bpe/devset.en.tok.bpe',
                           home + '/.dataset/latest4.esen.bpe/devset.es.matched.tok.bpe',
                           home + '/.dataset/latest4.esen.bpe/devset.en.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.esen.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.esen.bpe/train.en.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,
        'skip': 5000 / 32,
        'address': '147.8.182.14',

        # baseline models
        'baseline_xy': model + '/baseline_enes.npz',

        # test phase
        'trans_from': home + '/.dataset/latest4.esen.bpe/devset.es.tok.bpe',
        'tm_source':  home + '/.dataset/latest4.esen.bpe/devset.es.matched.tok.bpe',
        'tm_target':  home + '/.dataset/latest4.esen.bpe/devset.en.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/latest4.esen.bpe/dev.enes.en.tok',
        'trans_to':   home + '/.translate/' + name + '.esen_bpe.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.enes/train.en.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.enes/train.es.top1.tok',
        'tm_rank': home + '/.dataset/top5k.enes/match_top100.pkl'
    }
    return config




def setup_enes_bpe_latest():
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7.latest4'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/latest3.enes.bpe/train.en.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/latest3.enes.bpe/train.es.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/latest3.enes.bpe/train.en.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/latest3.enes.bpe/train.es.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/latest3.enes.bpe/devset.en.tok.bpe',
                           home + '/.dataset/latest3.enes.bpe/devset.es.tok.bpe',
                           home + '/.dataset/latest3.enes.bpe/devset.en.matched.tok.bpe',
                           home + '/.dataset/latest3.enes.bpe/devset.es.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/latest3.enes.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest3.enes.bpe/train.es.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest3.enes.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest3.enes.bpe/train.es.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,
        'skip': 5000 / 32,
        'address': '147.8.182.14',

        # baseline models
        'baseline_xy': model + '/baseline_enes.npz',

        # test phase
        'trans_from': home + '/.dataset/latest3.enes.bpe/devset.en.tok.bpe',
        'tm_source':  home + '/.dataset/latest3.enes.bpe/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/latest3.enes.bpe/devset.es.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/latest3.enes.bpe/dev.enes.es.tok',
        'trans_to':   home + '/.translate/' + name + '.enes_bpe.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.enes/train.en.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.enes/train.es.top1.tok',
        'tm_rank': home + '/.dataset/top5k.enes/match_top100.pkl'
    }
    return config


def setup_ende_bpe_latest():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.latest4'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/latest4.ende.bpe/train.en.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/latest4.ende.bpe/train.de.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/latest4.ende.bpe/train.en.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/latest4.ende.bpe/train.de.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/latest4.ende.bpe/devset.en.tok.bpe',
                           home + '/.dataset/latest4.ende.bpe/devset.de.tok.bpe',
                           home + '/.dataset/latest4.ende.bpe/devset.en.matched.tok.bpe',
                           home + '/.dataset/latest4.ende.bpe/devset.de.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/latest4.ende.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.ende.bpe/train.de.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.ende.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/latest4.ende.bpe/train.de.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,
        'skip': 5000 / 32,
        'address': '147.8.182.14',

        # baseline models
        'baseline_xy': model + '/baseline_ende.npz',

        # test phase
        'trans_from': home + '/.dataset/latest4.ende.bpe/devset.en.tok.bpe',
        'tm_source':  home + '/.dataset/latest4.ende.bpe/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/latest4.ende.bpe/devset.de.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/latest4.ende.bpe/dev.ende.de.tok',
        'trans_to':   home + '/.translate/' + name + '.ende_bpe.dev.translate',

        # multi-tm test (not use)
        'tm_source_full': home + '/.dataset/top5k.ende/train.en.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.ende/train.es.top1.tok',
        'tm_rank': home + '/.dataset/top5k.ende/match_top100.pkl'
    }
    return config





def setup_fren():
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7'

    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok',          # source
                     home + '/.dataset/top5k.fren/train.en.top5.shuf.tok',          # target
                     home + '/.dataset/top5k.fren/train.fr.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/top5k.fren/train.en.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.fren/devset.fr.tok',
                           home + '/.dataset/top5k.fren/devset.en.tok',
                           home + '/.dataset/top5k.fren/devset.fr.matched.tok',
                           home + '/.dataset/top5k.fren/devset.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.en.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_fren.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.fren/devset.fr.tok',
        'tm_source':  home + '/.dataset/top5k.fren/devset.fr.matched.tok',
        'tm_target':  home + '/.dataset/top5k.fren/devset.en.matched.tok',
        'trans_ref':  home + '/.dataset/top5k.fren/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren/train.fr.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.fren/train.en.top1.tok',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'
    }
    return config


def setup_fren_bleu():
    # home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7'

    # home   = '/scratch/jg5223/exp/TMNMT'
    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok',          # source
                     home + '/.dataset/top5k.fren/train.en.top5.shuf.tok',          # target
                     home + '/.dataset/top5k.fren/train.fr.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/top5k.fren/train.en.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.fren.bleu/devset.fren.fr.tok',
                           home + '/.dataset/top5k.fren.bleu/devset.fren.en.tok',
                           home + '/.dataset/top5k.fren.bleu/devset.fren.fr.matched.tok',
                           home + '/.dataset/top5k.fren.bleu/devset.fren.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/top5k.fren/train.en.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_fren.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.fren.bleu/devset.fren.fr.tok',
        'tm_source':  home + '/.dataset/top5k.fren.bleu/devset.fren.fr.matched.tok',
        'tm_target':  home + '/.dataset/top5k.fren.bleu/devset.fren.en.matched.tok',
        'trans_ref':  home + '/.dataset/top5k.fren.bleu/devset.fren.en.tok',
        'trans_to':   home + '/.translate/' + name + '.bleu3.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren/train.fr.top1.tok',
        'tm_target_full': home + '/.dataset/top5k.fren/train.en.top1.tok',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'
    }
    return config



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
        'datasets': [home + '/.dataset/top5k.fren.bpe/train.fr.top5.shuf.tok.bpe',          # source
                     home + '/.dataset/top5k.fren.bpe/train.en.top5.shuf.tok.bpe',          # target
                     home + '/.dataset/top5k.fren.bpe/train.fr.top5.matched.shuf.tok.bpe',  # source-TM
                     home + '/.dataset/top5k.fren.bpe/train.en.top5.matched.shuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.fren.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/top5k.fren.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.fren.bpe/devset.fr.matched.tok.bpe',
                           home + '/.dataset/top5k.fren.bpe/devset.en.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/top5k.fren.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.fren.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.fren.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.fren.bpe/train.en.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,

        # baseline models
        'baseline_xy': model + '/baseline_fren.bpe.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.fren.bpe/devset.fr.tok.bpe',
        'tm_source':  home + '/.dataset/top5k.fren.bpe/devset.fr.matched.tok.bpe',
        'tm_target':  home + '/.dataset/top5k.fren.bpe/devset.en.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.fren/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren.bpe/train.fr.top1.tok.bpe',
        'tm_target_full': home + '/.dataset/top5k.fren.bpe/train.en.top1.tok.bpe',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'

    }
    return config


def setup_fren_bpe_28():
    home = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    name  = 'TM2.B7.bpe2'

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

        # baseline models
        'baseline_xy': model + '/baseline_fren.bpe.npz',

        # test phase
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






def setup_enfr_bpe():
    home  = '/root/workspace/TMNMT'
    model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B7.bpe.test'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.enfr.bpe/train.en.top5.reshuf.tok.bpe',          # source
                     home + '/.dataset/top5k.enfr.bpe/train.fr.top5.reshuf.tok.bpe',          # target
                     home + '/.dataset/top5k.enfr.bpe/train.en.top5.matched.reshuf.tok.bpe',  # source-TM
                     home + '/.dataset/top5k.enfr.bpe/train.fr.top5.matched.reshuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.enfr.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.en.matched.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.fr.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/top5k.enfr.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.fr.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,
        # 'skip': 4000/32,

        # baseline models
        'baseline_xy': model + '/baseline_enfr.bpe.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.enfr.bpe/devset.en.tok.bpe',
        'tm_source':  home + '/.dataset/top5k.enfr.bpe/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/top5k.enfr.bpe/devset.fr.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.fren/devset.fr.tok',
        'trans_to':   home + '/.translate/' + name + '.enfr.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren.bpe/train.en.top1.tok.bpe',
        'tm_target_full': home + '/.dataset/top5k.fren.bpe/train.fr.top1.tok.bpe',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'

    }
    return config


def setup_enfr_bpe_nyu():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    # home  = '/root/workspace/TMNMT'
    # model = '/root/disk/scratch/model-tmnmt'
    name  = 'TM2.B10.bpe'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/top5k.enfr.bpe/train.en.top5.reshuf.tok.bpe',          # source
                     home + '/.dataset/top5k.enfr.bpe/train.fr.top5.reshuf.tok.bpe',          # target
                     home + '/.dataset/top5k.enfr.bpe/train.en.top5.matched.reshuf.tok.bpe',  # source-TM
                     home + '/.dataset/top5k.enfr.bpe/train.fr.top5.matched.reshuf.tok.bpe'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/top5k.enfr.bpe/devset.en.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.fr.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.en.matched.tok.bpe',
                           home + '/.dataset/top5k.enfr.bpe/devset.fr.matched.tok.bpe'
                           ],

        'dictionaries': [home + '/.dataset/top5k.enfr.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.fr.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.en.top5.shuf.tok.bpe.pkl',
                         home + '/.dataset/top5k.enfr.bpe/train.fr.top5.shuf.tok.bpe.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 80,

        # baseline models
        'baseline_xy': model + '/baseline_enfr.bpe.npz',

        # test phase
        'trans_from': home + '/.dataset/top5k.enfr.bpe/devset.en.tok.bpe',
        'tm_source':  home + '/.dataset/top5k.enfr.bpe/devset.en.matched.tok.bpe',
        'tm_target':  home + '/.dataset/top5k.enfr.bpe/devset.fr.matched.tok.bpe',
        'trans_ref':  home + '/.dataset/top5k.enfr.bpe/devset.fr.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate',

        # multi-tm test
        'tm_source_full': home + '/.dataset/top5k.fren.bpe/train.en.top1.tok.bpe',
        'tm_target_full': home + '/.dataset/top5k.fren.bpe/train.fr.top1.tok.bpe',
        'tm_rank':   home + '/.dataset/top5k.fren/match_top100.pkl',
        'tm_record': home + '/.dataset/top5k.fren/match_record5.pkl'

    }
    return config



def setup_fren_cc():
    home  = '/home/thoma/work/TMNMT'
    model = '/home/thoma/scratch/tmnmt'
    name  = 'TM2.C3'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/cc.fren/train.fr.top5.shuf.tok',          # source
                     home + '/.dataset/cc.fren/train.en.top5.shuf.tok',          # target
                     home + '/.dataset/cc.fren/train.fr.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/cc.fren/train.en.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/cc.fren/devset.fr.tok',
                           home + '/.dataset/cc.fren/devset.en.tok',
                           home + '/.dataset/cc.fren/devset.fr.matched.tok',
                           home + '/.dataset/cc.fren/devset.en.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/cc.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/cc.fren/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/cc.fren/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/cc.fren/train.en.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_fren.npz',

        # test phase
        'trans_from': home + '/.dataset/cc.fren/devset.fr.tok',
        'tm_source':  home + '/.dataset/cc.fren/devset.fr.matched.tok',
        'tm_target':  home + '/.dataset/cc.fren/devset.en.matched.tok',
        'trans_ref':  home + '/.dataset/cc.fren/devset.en.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
    }
    return config




def setup_enfr():
    home  = '/home/thoma/work/TMNMT'
    model = '/home/thoma/scratch/tmnmt'
    #name  = 'TM2.v1'
    name  = 'TM2.A7'

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
        'trans_from': home + '/.dataset/tm2.fren/devset.enfr.en.tok',
        'tm_source':  home + '/.dataset/tm2.fren/devset.enfr.en.matched.tok',
        'tm_target':  home + '/.dataset/tm2.fren/devset.enfr.fr.matched.tok',
        'trans_ref':  home + '/.dataset/tm2.fren/devset.enfr.fr.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
    }
    return config


def setup_enfr_nyu():
    home  = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT'
    model = '/misc/kcgscratch1/ChoGroup/thoma_exp/memory/TMNMT/.model'
    #name  = 'TM2.v1'
    name  = 'TM2.B9'

    config = {
        # train phase
        'name': name,
        'saveto': model + '/' + name + '_',
        'datasets': [home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok',          # source
                     home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok',          # target
                     home + '/.dataset/tm2.enfr/train.en.top5.matched.shuf.tok',  # source-TM
                     home + '/.dataset/tm2.enfr/train.fr.top5.matched.shuf.tok'   # target-TM
                     ],

        'valid_datasets': [home + '/.dataset/tm2.enfr/devset.en.tok',
                           home + '/.dataset/tm2.enfr/devset.fr.tok',
                           home + '/.dataset/tm2.enfr/devset.en.matched.tok',
                           home + '/.dataset/tm2.enfr/devset.fr.matched.tok'
                           ],

        'dictionaries': [home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.en.top5.shuf.tok.pkl',
                         home + '/.dataset/tm2.enfr/train.fr.top5.shuf.tok.pkl'
                         ],

        'voc_sizes': [20000, 20000, 20000, 20000],
        'maxlen': 50,

        # baseline models
        'baseline_xy': model + '/baseline_enfr.bs64.npz',
        'nn_coverage':  True,

        # test phase
        'trans_from': home + '/.dataset/tm2.enfr/devset.en.tok',
        'tm_source':  home + '/.dataset/tm2.enfr/devset.en.matched.tok',
        'tm_target':  home + '/.dataset/tm2.enfr/devset.fr.matched.tok',
        'trans_ref':  home + '/.dataset/tm2.enfr/devset.fr.tok',
        'trans_to':   home + '/.translate/' + name + '.dev.translate'
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
