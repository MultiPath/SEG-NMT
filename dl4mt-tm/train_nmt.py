from nmt import *
from pprint import pprint
from setup import setup
from data_iterator import TextIterator, prepare_data, prepare_cross

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='fren')
args = parser.parse_args()

model_options = setup(args.m)
pprint(model_options)

# add random seed
model_options['rng']  = numpy.random.RandomState(seed=19920206)
model_options['trng'] = RandomStreams(model_options['rng'].randint(0, 2**32-1))
model_options['n_words_src'] = model_options['voc_sizes'][0]
model_options['n_words'] = model_options['voc_sizes'][1]


# load dictionaries and invert them
worddicts   = [None] * len(model_options['dictionaries'])
worddicts_r = [None] * len(model_options['dictionaries'])
for ii, dd in enumerate(model_options['dictionaries']):
    with open(dd, 'rb') as f:
        worddicts[ii] = pkl.load(f)
    worddicts_r[ii] = dict()
    for kk, vv in worddicts[ii].iteritems():
        worddicts_r[ii][vv] = kk

# reload options
if model_options['reload_'] and os.path.exists(model_options['saveto']):
    print 'Reloading model options'
    with open('%s.pkl' % model_options['saveto'], 'rb') as f:
        model_options = pkl.load(f)

        model_options['overwrite']  = False
        model_options['saveFreq']   = 500
        model_options['sampleFreq'] = 20

@Timeit
def build_networks(options):
    funcs = dict()

    print 'Building model: E -> F & F -> E model'
    params_ef = init_params(options, 'ef_')
    params_fe = init_params(options, 'fe_')
    print 'Done.'

    # reload parameters
    if options['reload_'] and os.path.exists(options['saveto']):
        print 'Reloading model parameters'
        params_ef = load_params(options['saveto'], params_ef)
        params_fe = load_params(options['saveto'], params_fe)

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

    print 'build forward-attention models (4 models simultaneously)...'
    ret_ef11 = build_model(tparams_ef, [x1, x1_mask, y1, y1_mask], options, 'ef_', False, True)   # E->F curr
    ret_fe11 = build_model(tparams_fe, [y1, y1_mask, x1, x1_mask], options, 'fe_', False, False)  # F->E curr
    ret_ef22 = build_model(tparams_ef, [x2, x2_mask, y2, y2_mask], options, 'ef_', False, True)   # E->F tm
    ret_fe22 = build_model(tparams_fe, [y2, y2_mask, x2, x2_mask], options, 'fe_', False, False)  # F->E tm

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

    v1 = ret_ef12['attention'].sum()
    v2 = ret_ef21['attention'].sum()
    v3 = ret_fe12['attention'].sum()
    v4 = ret_fe21['attention'].sum()

    print 'build gates!'
    params_gate  = OrderedDict()
    params_gate  = get_layer('bi')[0](options, params_gate, nin=2 * options['dim'])
    tparams_gate = init_tparams(params_gate)

    if options['build_gate']:
        def build_gate(ctx1, ctx2):
            return get_layer('bi')[1](tparams_gate, ctx1, ctx2)

        gate_ef1 = 1 - build_gate(ret_ef11['ctxs'], ret_ef12['ctxs'])
        gate_ef2 = 1 - build_gate(ret_ef22['ctxs'], ret_ef21['ctxs'])
        gate_fe1 = 1 - build_gate(ret_fe11['ctxs'], ret_fe12['ctxs'])
        gate_fe2 = 1 - build_gate(ret_fe22['ctxs'], ret_fe21['ctxs'])

        print 'Building Gate functions, ...',
        f_gate = theano.function([ret_ef11['ctxs'], ret_ef12['ctxs']],
                                  gate_ef1, profile=profile)
        print 'Done.'

    else:
        print 'Building a Natural Gate Function'
        gate_ef1 = 1 - tensor.clip(ret_ef12['att_sum'] / (ret_ef11['att_sum'] + ret_ef12['att_sum']), 0, 1)
        gate_ef2 = 1 - tensor.clip(ret_ef21['att_sum'] / (ret_ef22['att_sum'] + ret_ef21['att_sum']), 0, 1)
        gate_fe1 = 1 - tensor.clip(ret_fe12['att_sum'] / (ret_fe11['att_sum'] + ret_fe12['att_sum']), 0, 1)
        gate_fe2 = 1 - tensor.clip(ret_fe21['att_sum'] / (ret_fe22['att_sum'] + ret_fe21['att_sum']), 0, 1)

    print 'build loss function (w/o gate)'

    # get the loss function
    def compute_prob(probs, y, y_mask):

        # compute the loss for the vocabulary-selection side
        y_flat  = y.flatten()
        n_words = probs.shape[-1]
        y_flat_idx = tensor.arange(y_flat.shape[0]) * n_words + y_flat
        probw   = probs.flatten()[y_flat_idx]
        probw   = probw.reshape([y.shape[0], y.shape[1]]) * y_mask
        return probw

    prob_ef11 = ret_ef11['probs']
    prob_ef22 = ret_ef22['probs']
    prob_fe11 = ret_fe11['probs']
    prob_fe22 = ret_fe22['probs']

    def compute_cost(prob, y, y_mask, att, t, t_mask, g):
        _y = tensor.eq(y, 1)
        y_mask *= ((1 - _y) + _y * (1 - t_mask))
        ccost = -tensor.log(compute_prob(prob, y, y_mask) * g +
                            compute_prob(att, t, t_mask) * (1 - g) +
                            1e-7)
        ccost = (ccost * (1 - (1 - y_mask) * (1 - t_mask))).sum(0)
        return ccost

    # get cost
    cost_ef1 = compute_cost(prob_ef11, y1, y1_mask, att_ef12, tef12, tef12_mask, gate_ef1)
    cost_ef2 = compute_cost(prob_ef22, y2, y2_mask, att_ef21, tef21, tef21_mask, gate_ef2)
    cost_fe1 = compute_cost(prob_fe11, x1, x1_mask, att_fe12, tfe12, tfe12_mask, gate_fe1)
    cost_fe2 = compute_cost(prob_fe22, x2, x2_mask, att_fe21, tfe21, tfe21_mask, gate_fe2)

    cost  = cost_ef1 + cost_ef2 + cost_fe1 + cost_fe2
    value = v1 + v2 + v3 + v4

    print 'build sampler (one-step)'
    f_init_ef, f_next_ef = build_sampler(tparams_ef, options, options['trng'], 'ef_')
    f_init_fe, f_next_fe = build_sampler(tparams_fe, options, options['trng'], 'fe_')

    print 'build attender (one-step)'
    f_attend_ef = build_attender(tparams_ef, None, options, 'ef_', one_step=True)  # E->F curr
    f_attend_fe = build_attender(tparams_fe, None, options, 'fe_', one_step=True)

    # before any regularizer
    print 'build Cost Function...',
    inputs = [x1, x1_mask, y1, y1_mask, x2, x2_mask, y2, y2_mask,
              tef12, tef12_mask, tef21, tef21_mask,
              tfe12, tfe12_mask, tfe21, tfe21_mask]
    f_valid = theano.function(inputs, cost, profile=profile)

    print 'build Gradient (backward)...',
    cost    = cost.mean()

    if options['build_gate']:
        tparams = dict(tparams_ef.items() + tparams_fe.items() + tparams_gate.items())
    else:
        tparams = dict(tparams_ef.items() + tparams_fe.items())

    grads   = clip(tensor.grad(cost, wrt=itemlist(tparams)), options['clip_c'])
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building Optimizers...',
    f_cost, f_update = eval(options['optimizer'])(
        lr, tparams, grads, inputs, [cost, value])

    print 'Done'

    # put everything into function lists
    funcs['valid']  = f_valid
    funcs['cost']   = f_cost
    funcs['update'] = f_update

    funcs['init_ef'] = f_init_ef
    funcs['init_fe'] = f_init_fe
    funcs['next_ef'] = f_next_ef
    funcs['next_fe'] = f_next_fe

    funcs['att_ef']  = f_attend_ef
    funcs['att_fe']  = f_attend_fe

    funcs['crit_ef'] = ret_ef11['f_critic']
    funcs['crit_fe'] = ret_ef22['f_critic']

    if options['build_gate']:
        funcs['gate']    = f_gate

    print 'Build Networks... done!'
    return funcs, tparams

funcs, tparams = build_networks(model_options)

# print 'save the compiled functions/tparams for temperal usage'


print 'Loading data'
train = TextIterator(model_options['datasets'], model_options['dictionaries'], [0, 0, 0, 0],
                     batch_size=model_options['batch_size'], maxlen=model_options['maxlen'])
valid = TextIterator(model_options['valid_datasets'], model_options['dictionaries'], [0, 0, 0, 0],
                     batch_size=model_options['batch_size'], maxlen=200)

if model_options['use_pretrain']:
    print 'use the pretrained NMT-models...',
    params = unzip(tparams)
    params = load_params2(model_options['baseline_ef'], params, mode='ef_')
    params = load_params2(model_options['baseline_fe'], params, mode='fe_')
    zipp(params, tparams)
    print 'Done.'

else:
    print 'not loading the pretrained baseline'

print '-------------------------------------------- Main-Loop -------------------------------------------------'

# ------------------ initlization --------------- #
best_p       = None
bad_counter  = 0
uidx         = 0
estop        = False
history_errs = []
max_epochs   = 100
finish_after = 10000000

lrate        = model_options['lrate']
saveFreq     = model_options['saveFreq']
sampleFreq   = model_options['sampleFreq']
validFreq    = model_options['validFreq']
saveto       = model_options['saveto']
overwrite    = model_options['overwrite']

# ----------------------------------------------- #

# reload history
if model_options['reload_'] and os.path.exists(model_options['saveto']):
    rmodel = numpy.load(model_options['saveto'])
    history_errs = list(rmodel['history_errs'])
    if 'uidx' in rmodel:
        uidx = rmodel['uidx']


# idx back to sequences
def idx2seq(x, ii):
    seq = []
    for vv in x:
        if vv == 0:
            break
        if vv in worddicts_r[ii]:
            seq.append(worddicts_r[ii][vv])
        else:
            seq.append('UNK')
    return ' '.join(seq)


# compute-update
@Timeit
def execute(inps, lrate, info):
    eidx, uidx = info
    cost, value = funcs['cost'](*inps)
    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Value', value

    # check for bad numbers, usually we remove non-finite elements
    # and continue training - but not done here
    if numpy.isnan(cost):
        raise Exception('Cost NaN detected')

    if numpy.isinf(cost):
        raise Exception('Cost Inf detected')

    funcs['update'](lrate)

    return cost


@Timeit
def validate(funcs, options, iterator, verbose=False):
    probs = []

    n_done = 0
    for k, (sx1, sy1, sx2, sy2) in enumerate(iterator):
        x1, x1_mask = prepare_data(sx1, 200, options['voc_sizes'][0])
        y1, y1_mask = prepare_data(sy1, 200, options['voc_sizes'][1])
        x2, x2_mask = prepare_data(sx2, 200, options['voc_sizes'][2])
        y2, y2_mask = prepare_data(sy2, 200, options['voc_sizes'][3])

        print 'x1:{}, x2:{}, y1:{}, y2:{}'.format(x1.shape, x2.shape, y1.shape, y2.shape)

        tx12, tx12_mask = prepare_cross(sx1, sx2, x1.shape[0])
        tx21, tx21_mask = prepare_cross(sx2, sx1, x2.shape[0])
        ty12, ty12_mask = prepare_cross(sy1, sy2, y1.shape[0])
        ty21, ty21_mask = prepare_cross(sy1, sy2, y2.shape[0])

        inps = [x1, x1_mask, y1, y1_mask,
                x2, x2_mask, y2, y2_mask,
                ty12, ty12_mask, ty21, ty21_mask,
                tx12, tx12_mask, tx21, tx21_mask]

        pprobs = funcs['valid'](*inps)
        for pp in pprobs:
            probs.append(pp)

        # if numpy.isnan(numpy.mean(probs)):
        #     ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# start!!
for eidx in xrange(max_epochs):
    n_samples = 0

    for k, (sx1, sy1, sx2, sy2) in enumerate(train):
        uidx += 1

        x1, x1_mask = prepare_data(sx1, model_options['maxlen'], model_options['voc_sizes'][0])
        y1, y1_mask = prepare_data(sy1, model_options['maxlen'], model_options['voc_sizes'][1])
        x2, x2_mask = prepare_data(sx2, model_options['maxlen'], model_options['voc_sizes'][2])
        y2, y2_mask = prepare_data(sy2, model_options['maxlen'], model_options['voc_sizes'][3])

        tx12, tx12_mask = prepare_cross(sx1, sx2, x1.shape[0])
        tx21, tx21_mask = prepare_cross(sx2, sx1, x2.shape[0])
        ty12, ty12_mask = prepare_cross(sy1, sy2, y1.shape[0])
        ty21, ty21_mask = prepare_cross(sy1, sy2, y2.shape[0])

        print 'x1:{}, x2:{}, y1:{}, y2:{}'.format(x1.shape, x2.shape, y1.shape, y2.shape)

        inps = [x1, x1_mask, y1, y1_mask,
                x2, x2_mask, y2, y2_mask,
                ty12, ty12_mask, ty21, ty21_mask,
                tx12, tx12_mask, tx21, tx21_mask]

        try:
            execute(inps, lrate, [eidx, uidx])  # train one step.

        except Exception, e:
            print e
            continue

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
            for jj in xrange(numpy.minimum(5, x1.shape[1])):
                stochastic = True
                sample, sc, acts = gen_sample(tparams, funcs,
                                           x1[:, jj][:, None],
                                           x2[:, jj][:, None],
                                           y2[:, jj][:, None],
                                           model_options,
                                           rng=model_options['rng'],
                                           m=1,
                                           k=1,
                                           maxlen=200,
                                           stochastic=model_options['stochastic'],
                                           argmax=True)

                print 'Source-CR {}: {}'.format(jj, idx2seq(sx1[jj], 0))
                print 'Target-CR {}: {}'.format(jj, idx2seq(sy1[jj], 1))
                print '-----------------------------'
                print 'Source-TM {}: {}'.format(jj, idx2seq(sx2[jj], 2))
                print 'Target-TM {}: {}'.format(jj, idx2seq(sy2[jj], 3))
                print '============================='

                if model_options['stochastic']:
                    ss = sample
                else:
                    sc /= numpy.array([len(s) for s in sample])
                    ss = sample[sc.argmin()]

                _ss = []
                for ii, si in enumerate(ss):
                    if si < model_options['voc_sizes'][1]:
                        _ss.append(si)
                    else:
                        print si
                        offset = si - model_options['voc_sizes'][1]
                        _ss.append(sy2[jj][offset])

                print 'Sample-CR {}: {}'.format(jj, idx2seq(_ss, 1))
                print 'Copy Prob {}: {}'.format(jj, ' '.join(['{:.2f}'.format(a) for a in acts]))
                print

        # validate model on validation set and early stop if necessary
        # if numpy.mod(uidx, validFreq) == 0:
        #    # use_noise.set_value(0.)
        #    valid_errs = validate(funcs, model_options, valid, False)
        #    valid_err  = valid_errs.mean()
        #    history_errs.append(valid_err)

        #    if numpy.isnan(valid_err):
        #        print 'NaN detected'
        #        sys.exit(-1)

        #    print 'Valid ', valid_err

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

valid_err = validate(funcs, model_options, valid).mean()
print 'Valid ', valid_err

params = copy.copy(best_p)
numpy.savez(saveto, zipped_params=best_p,
            history_errs=history_errs,
            uidx=uidx,
            **params)



