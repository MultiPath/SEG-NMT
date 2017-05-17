'''
Translates a source file using a translation model.
'''
import argparse
import theano
import numpy
import time
import os
import itertools
import cPickle as pkl
from layer import *
from nmt import (build_sampler, gen_sample, gen_sample_multi, load_params,
                 init_params, init_tparams, build_networks)
from setup import setup


def translate_model(queue, funcs, tparams, options, k,
                    normalize, m=0, d_maxlen=200, mm=0):

    use_noise = theano.shared(numpy.float32(0.))
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(19920206)

    def _translate(seq_x1, seq_x2=None, seq_y2=None):

        if mm == 0:   # don't use translation memory.
            sample, score = \
                gen_sample(tparams, funcs['init_xy'], funcs['next_xy'],
                           numpy.array(seq_x1).reshape([len(seq_x1), 1]),
                           options, rng=trng, k=k, maxlen=d_maxlen,
                           stochastic=options['stochastic'], argmax=True)

            action = [0 for _ in score]
            gating = [0 for _ in score]

        else:
            # sample given an input sequence and obtain scores
            sample, score, action, gating = \
                gen_sample_multi(tparams, funcs,
                                  numpy.array(seq_x1).reshape([len(seq_x1), 1]),
                                  [numpy.array(seq_x20).reshape([len(seq_x20), 1]) for seq_x20 in seq_x2],
                                  [numpy.array(seq_y20).reshape([len(seq_y20), 1]) for seq_y20 in seq_y2],
                                  options, rng=trng, m=m, k=k, maxlen=d_maxlen,
                                  stochastic=options['stochastic'], argmax=True)

        # normalize scores according to sequence lengths
        if k > 1:
            if normalize:
                lengths = numpy.array([len(s) for s in sample])
                score  /= lengths
                # score  /= (lengths ** 0.7)

            sidx   = numpy.argmin(score)
            sample, score, action, gating = \
                    sample[sidx], score[sidx], action[sidx], gating[sidx]

        return sample, score, action, gating

    rqueue = []
    time1  = time.time()
    for req in queue:
        if mm != 0:
            idx, sx1, sx2, sy2 = req[0], req[1], req[2], req[3]
            x1 = map(lambda ii: ii if ii < options['voc_sizes'][0] else 1, sx1)
            x2 = [map(lambda ii: ii if ii < options['voc_sizes'][2] else 1, sx20) for sx20 in sx2]
            y2 = [map(lambda ii: ii if ii < options['voc_sizes'][3] else 1, sy20) for sy20 in sy2]
            seq, ss, acts, gs = _translate(x1, x2, y2)

            ssy2 = list(itertools.chain.from_iterable(sy2))
            sseq = map(lambda ii: ii if ii < options['voc_sizes'][1] else ssy2[ii-options['voc_sizes'][1]], seq)

        else:
            idx, sx1  = req[0], req[1]
            x1   = map(lambda ii: ii if ii < options['voc_sizes'][0] else 1, sx1)
            seq, ss, acts, gs = _translate(x1)
            sseq = map(lambda ii: ii if ii < options['voc_sizes'][1] else 1, seq)

        if idx % 10 == 0:
            print '[test] complete:{}, {}s'.format(idx, time.time() - time1)

        rqueue.append((sseq, ss, acts, gs))

    print '[test] complete:{}, {}s'.format(idx, time.time() - time1)
    return rqueue


def go(model, dictionary, dictionary_target,
       source_file_x1, source_file_x2, source_file_y2, tm_rank,
       reference_file_y1, saveto,
       k=5, normalize=False, d_maxlen=200, MM=1, iters=-1, SS=False):

    # inter-step
    step_test = 0

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)
        options['see_pretrain'] = False

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    print 'load rank file...{}'.format(tm_rank),
    if MM < 0:
        print 'adaptive translation memory'
        ret = pkl.load(open(tm_rank, 'r'))
        ranks, recalls = ret[0], ret[1]
    else:
        ranks = pkl.load(open(tm_rank, 'r'))

    print 'load full_source...',
    source_set = open(source_file_x2, 'r').readlines()

    print 'load full target...',
    target_set = open(source_file_y2, 'r').readlines()

    print 'done.'

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname_x1):
        queue_x1 = []
        queue_x2 = []
        queue_y2 = []
        queue = []
        with open(fname_x1, 'r') as f:
            for idx, line in enumerate(f):

                words = line.strip().split()
                x1 = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x1 += [0]
                queue_x1.append((idx, x1))

        if MM != 0:

            for idx in range(len(ranks)):
                x2s = []
                if MM > 0:
                    L = MM
                else:
                    L = len(ranks[idx])

                for jdx in range(L):
                    words = source_set[ranks[idx][jdx]].strip().split()
                    x2 = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                    x2 += [0]
                    x2s.append(x2)
                queue_x2.append((idx, x2s))

            for idx in range(len(ranks)):
                y2s = []
                if MM > 0:
                    L = MM
                else:
                    L = len(ranks[idx])

                for jdx in range(L):
                    words = target_set[ranks[idx][jdx]].strip().split()
                    y2 = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                    y2 += [0]
                    y2s.append(y2)
                queue_y2.append((idx, y2s))

            for i, (x1, x2, y2) in enumerate(zip(queue_x1, queue_x2, queue_y2)):
                queue.append((i, x1[1], x2[1], y2[1]))

        else:
            queue = queue_x1

        return queue

    def _send_self(fname_x1, fname_y1):
        queue_x1 = []
        queue_x2 = []
        queue_y2 = []
        queue = []
        with open(fname_x1, 'r') as f:
            for idx, line in enumerate(f):

                words = line.strip().split()
                x1 = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x1 += [0]
                queue_x1.append((idx, x1))


        with open(fname_x1, 'r') as f:
            for idx, line in enumerate(f):

                words = line.strip().split()
                x1 = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x1 += [0]
                queue_x2.append((idx, [x1]))


        with open(fname_y1, 'r') as f:
            for idx, line in enumerate(f):

                words = line.strip().split()
                y1 = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                y1 += [0]
                queue_y2.append((idx, [y1]))

        for i, (x1, x2, y2) in enumerate(zip(queue_x1, queue_x2, queue_y2)):
            queue.append((i, x1[1], x2[1], y2[1]))

        return queue


    if iters > -1:
        model = model[:-4] + '.iter{}.npz'.format(iters)

    print '[test] build the model...{}'.format(model)
    funcs, tparams = build_networks(options, model, train=False)

    if not SS:
        saveto = saveto + '-mm=' + str(MM) + '.multi2'
        queue = _send_jobs(source_file_x1)
    else:
        saveto = saveto + '.multi.SSR'
        queue = _send_self(source_file_x1, reference_file_y1)

    print '[test] start translating ', source_file_x1, '...to...', saveto
    rets  = translate_model(queue, funcs, tparams, options,
                            k, normalize, 0, d_maxlen, MM)
    sseqs, ss, acts, gs = zip(*rets)

    trans = _seqs2words(sseqs)
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'

    pkl.dump(rets, open(saveto + '.pkl', 'w'))
    print 'All Done'




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='fren')
    parser.add_argument('-p', type=str, default='test')
    parser.add_argument('-mm', default=0)
    parser.add_argument('-i',  default=-1)
    parser.add_argument('-ss', action='store_true', default=False)
    args = parser.parse_args()
    args.mm = int(args.mm)
    config  = setup(args.m)

    if args.mm < 0:  # adaptive mode.
        tm_rank = config['tm_record']
    else:
        tm_rank = config['tm_rank']
    print tm_rank

    # import sys; sys.exit(1)
    if args.p == 'round':
        print 'ROUND-MODE'
        go(config['saveto'],
           config['dictionaries'][0],
           config['dictionaries'][1],
           config['trans_from'],
           config['tm_source'],
           config['tm_target'],
           config['trans_ref'],
           config['trans_to'],
           config['beamsize'],
           config['normalize'],
           config['d_maxlen'])

    else:
        print 'TEST-MODE'

        go(config['saveto'],
           config['dictionaries'][0],
           config['dictionaries'][1],
           config['trans_from'],
           config['tm_source_full'],
           config['tm_target_full'],
           tm_rank,
           config['trans_ref'],
           config['trans_to'],
           config['beamsize'],
           config['normalize'],
           config['d_maxlen'],
           MM=args.mm,
           iters=args.i,
           SS=args.ss)

    print 'all done'
