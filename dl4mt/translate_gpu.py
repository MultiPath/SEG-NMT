'''
Translates a source file using a translation model.
'''
import argparse
import theano
import numpy
import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)
from setup import setup


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(19920206)

def translate_model(queue, model, options, k, normalize):

    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    rqueue = []
    for req in queue:
        idx, x = req[0], req[1]
        print 'translate-', idx
        seq = _translate(x)
        rqueue.append(seq)

    return rqueue


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False,chr_level=False, *args, **kwargs):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

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

    def _send_jobs(fname):
        queue = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x += [0]
                queue.append((idx, x))
        return queue


    print 'Translating ', source_file, '...'
    queue = _send_jobs(source_file)
    trans = _seqs2words(translate_model(queue, model, options, k, normalize))
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='fren')
    args = parser.parse_args()

    config = setup(args.m)

    main(config['saveto'],
         config['dictionaries'][0],
         config['dictionaries'][1],
         config['trans_from'],
         config['trans_to'],
         config['beamsize'],
         config['normalize'], False)

    print 'all done'
