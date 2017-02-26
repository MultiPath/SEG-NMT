from nmt import *
from pprint import pprint
from setup import setup
from data_iterator import TextIterator, prepare_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='fren')
args = parser.parse_args()

config = setup(args.m)
pprint(config)


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq

          datasets=None,
          valid_datasets=None,
          dictionaries=None,
          voc_sizes=None,

          use_dropout=False,
          reload_=False,
          overwrite=False,
          run_BLEU=False,
          *args, **kwargs):

    # Model options
    model_options = locals().copy()

    # add random seed
    model_options['trng'] = RandomStreams(19920206)
    model_options['n_words_src'] = model_options['voc_sizes'][0]
    model_options['n_words'] = model_options['voc_sizes'][1]

    # load dictionaries and invert them
    worddicts   = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets, dictionaries, voc_sizes, batch_size=batch_size, maxlen=maxlen)
    valid = TextIterator(valid_datasets, dictionaries,voc_sizes, batch_size=valid_batch_size, maxlen=200)

    @Timeit
    def build_networks(options):
        funcs = dict()

        print 'Building model: E -> F & F -> E model'
        params_ef = init_params(options, 'ef_')
        params_fe = init_params(options, 'fe_')
        print 'Done.'

        # reload parameters
        if reload_ and os.path.exists(saveto):
            print 'Reloading model parameters'
            params_ef = load_params(saveto, params_ef)
            params_fe = load_params(saveto, params_fe)

        tparams_ef = init_tparams(params_ef)
        tparams_fe = init_tparams(params_fe)

        # inputs of the model (x1, y1, x2, y2)
        x1 = tensor.matrix('x1', dtype='int64')
        x1_mask = tensor.matrix('x1_mask', dtype='float32')
        y1 = tensor.matrix('y1', dtype='int64')
        y1_mask = tensor.matrix('y1_mask', dtype='float32')
        x2 = tensor.matrix('x2', dtype='int64')
        x2_mask = tensor.matrix('x2_mask', dtype='float32')
        y2 = tensor.matrix('y2', dtype='int64')
        y2_mask = tensor.matrix('y2_mask', dtype='float32')

        # TM reference index
        tef12 = tensor.matrix('ef12', dtype='int64')
        tef12_mask = tensor.matrix('ef12_mask', dtype='float32')
        tef21 = tensor.matrix('ef21', dtype='int64')
        tef21_mask = tensor.matrix('ef21_mask', dtype='float32')
        tfe12 = tensor.matrix('fe12', dtype='int64')
        tfe12_mask = tensor.matrix('fe12_mask', dtype='float32')
        tfe21 = tensor.matrix('fe21', dtype='int64')
        tfe21_mask = tensor.matrix('fe21_mask', dtype='float32')

        print 'build forward-attention models (4 models simultaneously)'
        ret_ef11 = build_model(tparams_ef, [x1, x1_mask, y1, y1_mask], options, 'ef_', False)  # E->F curr
        ret_fe11 = build_model(tparams_fe, [y1, y1_mask, x1, x1_mask], options, 'fe_', False)  # F->E curr
        ret_ef22 = build_model(tparams_ef, [x2, x2_mask, y2, y2_mask], options, 'ef_', False)  # E->F tm
        ret_fe22 = build_model(tparams_fe, [y2, y2_mask, x2, x2_mask], options, 'fe_', False)  # F->E tm

        print 'build cross-attention models'
        ret_ef12 = build_attender(tparams_ef,
                                  [ret_ef11['prev_hids'], ret_ef11['prev_emb'], ret_ef22['ctx'], x2_mask],
                                  options, 'ef_')  # E->F curr
        ret_ef21 = build_attender(tparams_ef,
                                  [ret_ef22['prev_hids'], ret_ef22['prev_emb'], ret_ef11['ctx'], x1_mask],
                                  options, 'ef_')  # E->F tm
        ret_fe12 = build_attender(tparams_fe,
                                  [ret_fe11['prev_hids'], ret_fe11['prev_emb'], ret_fe22['ctx'], y2_mask],
                                  options, 'fe_')  # F->E curr
        ret_fe21 = build_attender(tparams_fe,
                                  [ret_fe22['prev_hids'], ret_fe22['prev_emb'], ret_fe11['ctx'], y1_mask],
                                  options, 'fe_')  # F->E tm

        print 'build attentions (forward, cross-propagation)'

        def build_prop(atten_ef, atten_fe):
            atten_ef = atten_ef.dimshuffle(1, 0, 2)
            atten_fe = atten_fe.dimshuffle(1, 0, 2)
            attention = tensor.batched_dot(atten_ef, atten_fe).dimshuffle(1, 0, 2)
            return attention

        att_ef12 = build_prop(ret_ef12['attention'], ret_fe22['attention'])
        att_ef21 = build_prop(ret_ef21['attention'], ret_fe11['attention'])
        att_fe12 = build_prop(ret_fe12['attention'], ret_ef22['attention'])
        att_fe21 = build_prop(ret_fe21['attention'], ret_ef11['attention'])

        print 'build loss function (w/o gate)'

        # we first try the simplest version: use a natural attention-gate.
        # TODO: make it as a Neural Gate
        gate_ef1 = ret_ef11['att_sum'] / (ret_ef11['att_sum'] + ret_ef12['att_sum'])
        gate_ef2 = ret_ef22['att_sum'] / (ret_ef22['att_sum'] + ret_ef21['att_sum'])
        gate_fe1 = ret_fe11['att_sum'] / (ret_fe11['att_sum'] + ret_fe12['att_sum'])
        gate_fe2 = ret_fe22['att_sum'] / (ret_fe22['att_sum'] + ret_fe21['att_sum'])

        # get the loss function
        def compute_prob(probs, y, y_mask):

            # compute the loss for the vocabulary-selection side
            y_flat = y.flatten()
            y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
            probw = probs.flatten()[y_flat_idx]
            probw = probw.reshape([y.shape[0], y.shape[1]]) * y_mask
            return probw

        prob_ef11 = ret_ef11['probs']
        prob_ef22 = ret_ef22['probs']
        prob_fe11 = ret_fe11['probs']
        prob_fe22 = ret_fe22['probs']

        # get cost
        cost_ef1 = (-tensor.log(compute_prob(prob_ef11, y1, y1_mask) * gate_ef1 +
                                compute_prob(att_ef12, tef12, tef12_mask) * (1 - gate_ef1)
                                + 1e-8) * (1 - (1 - y1_mask) * (1 - tef12_mask))).sum(0)
        cost_ef2 = (-tensor.log(compute_prob(prob_ef22, y2, y2_mask) * gate_ef2 +
                                compute_prob(att_ef21, tef21, tef21_mask) * (1 - gate_ef2)
                                + 1e-8) * (1 - (1 - y2_mask) * (1 - tef21_mask))).sum(0)
        cost_fe1 = (-tensor.log(compute_prob(prob_fe11, x1, x1_mask) * gate_fe1 +
                                compute_prob(att_fe12, tfe12, tfe12_mask) * (1 - gate_fe1)
                                + 1e-8) * (1 - (1 - x1_mask) * (1 - tfe12_mask))).sum(0)
        cost_fe2 = (-tensor.log(compute_prob(prob_fe22, x2, x2_mask) * gate_fe2 +
                                compute_prob(att_fe21, tfe21, tfe21_mask) * (1 - gate_fe2)
                                + 1e-8) * (1 - (1 - x2_mask) * (1 - tfe21_mask))).sum(0)

        cost = cost_ef1 + cost_ef2 + cost_fe1 + cost_fe2

        # print 'Building sampler'
        # f_init, f_next = build_sampler(tparams, options, trng, use_noise)

        # before any regularizer
        print 'Building Cost Function...',
        inputs = [x1, x1_mask, y1, y1_mask, x2, x2_mask, y2, y2_mask,
                  tef12, tef12_mask, tef21, tef21_mask,
                  tfe12, tfe12_mask, tfe21, tfe21_mask]

        # f_cost = theano.function(inputs, cost, profile=profile)
        # print 'Done'

        cost = cost.mean()

        print 'Build Gradient (backward)...',

        tparams = dict(tparams_ef.items() + tparams_fe.items())
        grads   = clip(tensor.grad(cost, wrt=itemlist(tparams)), clip_c)
        print 'Done'

        # compile the optimizer, the actual computational graph is compiled here
        lr = tensor.scalar(name='lr')
        print 'Building Optimizers...',
        f_cost, f_update = eval(optimizer)(lr, tparams, grads, inputs, cost)

        funcs['cost']   = f_cost
        funcs['update'] = f_update

        print 'Done'
        return funcs

    # funcs = build_networks(model_options)

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0]) / batch_size

    # ***** special **** #
    BleuFreq = 2000
    # BleuPoint = 20000

    print '..Upto here.'
    import sys
    sys.exit(321)




    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1



            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # validate model with BLEU
            pass

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


validerr = train(**config)
print 'done'
