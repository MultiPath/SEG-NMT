from nmt import *
from pprint import pprint
from setup import setup
from data_iterator import TextIterator, prepare_data, prepare_cross
from termcolor import colored as clr

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

    print 'Building model: X -> Y & Y -> X model'
    params_xy = init_params(options, 'xy_')
    params_yx = init_params(options, 'yx_')
    print 'Done.'

    # use pre-trained models

    print 'load the pretrained NMT-models...',
    params_xy = load_params2(model_options['baseline_xy'], params_xy, mode='xy_')
    params_yx = load_params2(model_options['baseline_yx'], params_yx, mode='yx_')
    tparams_xy0 = init_tparams(params_xy)  # pre-trained E->F model
    tparams_yx0 = init_tparams(params_yx)  # pre-trained F->E model
    print 'Done.'

    # reload parameters
    if options['reload_'] and os.path.exists(options['saveto']):
        print 'Reloading model parameters'
        params_xy = load_params(options['saveto'], params_xy)
        params_yx = load_params(options['saveto'], params_yx)

    tparams_xy = init_tparams(params_xy)
    tparams_yx = init_tparams(params_yx)

    # inputs of the model (x1, y1, x2, y2)
    x1 = tensor.matrix('x1', dtype='int64')
    x1_mask = tensor.matrix('x1_mask', dtype='float32')
    y1 = tensor.matrix('y1', dtype='int64')
    y1_mask = tensor.matrix('y1_mask', dtype='float32')
    x2 = tensor.matrix('x2', dtype='int64')
    x2_mask = tensor.matrix('x2_mask', dtype='float32')
    y2 = tensor.matrix('y2', dtype='int64')
    y2_mask = tensor.matrix('y2_mask', dtype='float32')

    # TM rxyerence index
    txy12 = tensor.matrix('xy12', dtype='int64')
    txy12_mask = tensor.matrix('xy12_mask', dtype='float32')
    txy21 = tensor.matrix('xy21', dtype='int64')
    txy21_mask = tensor.matrix('xy21_mask', dtype='float32')
    tyx12 = tensor.matrix('yx12', dtype='int64')
    tyx12_mask = tensor.matrix('yx12_mask', dtype='float32')
    tyx21 = tensor.matrix('yx21', dtype='int64')
    tyx21_mask = tensor.matrix('yx21_mask', dtype='float32')

    print 'build forward-attention models (4 models simultaneously)...'
    ret_xy11 = build_model(tparams_xy, [x1, x1_mask, y1, y1_mask], options, 'xy_', False, True)   # E->F curr
    ret_yx11 = build_model(tparams_yx, [y1, y1_mask, x1, x1_mask], options, 'yx_', False, True)  # F->E curr
    ret_xy22 = build_model(tparams_xy, [x2, x2_mask, y2, y2_mask], options, 'xy_', False, False)   # E->F tm
    ret_yx22 = build_model(tparams_yx, [y2, y2_mask, x2, x2_mask], options, 'yx_', False, False)  # F->E tm

    print 'build cross-attention models'
    ret_xy12 = build_attender(tparams_xy,
                              [ret_xy11['prev_hids'], ret_xy11['prev_emb'], ret_xy22['ctx'], x2_mask],
                              options, 'xy_')  # E->F curr
    ret_xy21 = build_attender(tparams_xy,
                              [ret_xy22['prev_hids'], ret_xy22['prev_emb'], ret_xy11['ctx'], x1_mask],
                              options, 'xy_')  # E->F tm
    ret_yx12 = build_attender(tparams_yx,
                              [ret_yx11['prev_hids'], ret_yx11['prev_emb'], ret_yx22['ctx'], y2_mask],
                              options, 'yx_')  # F->E curr
    ret_yx21 = build_attender(tparams_yx,
                              [ret_yx22['prev_hids'], ret_yx22['prev_emb'], ret_yx11['ctx'], y1_mask],
                              options, 'yx_')  # F->E tm

    print 'build attentions (forward, cross-propagation)'

    def build_prop(atten_xy, atten_yx):
        atten_xy = atten_xy.dimshuffle(1, 0, 2)
        atten_yx = atten_yx.dimshuffle(1, 0, 2)
        attention = tensor.batched_dot(atten_xy, atten_yx).dimshuffle(1, 0, 2)
        return attention

    att_xy12 = build_prop(ret_xy12['attention'], ret_yx22['attention'])
    att_xy21 = build_prop(ret_xy21['attention'], ret_yx11['attention'])
    att_yx12 = build_prop(ret_yx12['attention'], ret_xy22['attention'])
    att_yx21 = build_prop(ret_yx21['attention'], ret_xy11['attention'])

    print 'build gates!'
    params_gate  = OrderedDict()
    params_gate  = get_layer('bi')[0](options, params_gate,
                                      nin1=options['dim'],
                                      nin2=2 * options['dim'])
    tparams_gate = init_tparams(params_gate)

    if options['build_gate']:
        def build_gate(hx1, ctx1, ctx2):
            v1 = get_layer('bi')[1](tparams_gate, hx1, ctx1, activ='lambda x: tensor.tanh(x)')
            v2 = get_layer('bi')[1](tparams_gate, hx1, ctx2, activ='lambda x: tensor.tanh(x)')
            return tensor.nnet.sigmoid(v1 - v2)

        gate_xy1 = build_gate(ret_xy11['hids'], ret_xy11['ctxs'], ret_xy12['ctxs'])
        gate_xy2 = build_gate(ret_xy22['hids'], ret_xy22['ctxs'], ret_xy21['ctxs'])
        gate_yx1 = build_gate(ret_yx11['hids'], ret_yx11['ctxs'], ret_yx12['ctxs'])
        gate_yx2 = build_gate(ret_yx22['hids'], ret_yx22['ctxs'], ret_yx21['ctxs'])

        print 'Building Gate functions, ...',
        f_gate = theano.function([ret_xy11['hids'], ret_xy11['ctxs'], ret_xy12['ctxs']],
                                  gate_xy1, profile=profile)
        print 'Done.'

    else:
        print 'Building a Natural Gate Function'
        gate_xy1 = 1 - tensor.clip(ret_xy12['att_sum'] / (ret_xy11['att_sum'] + ret_xy12['att_sum']), 0, 1)
        gate_xy2 = 1 - tensor.clip(ret_xy21['att_sum'] / (ret_xy22['att_sum'] + ret_xy21['att_sum']), 0, 1)
        gate_yx1 = 1 - tensor.clip(ret_yx12['att_sum'] / (ret_yx11['att_sum'] + ret_yx12['att_sum']), 0, 1)
        gate_yx2 = 1 - tensor.clip(ret_yx21['att_sum'] / (ret_yx22['att_sum'] + ret_yx21['att_sum']), 0, 1)

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

    prob_xy11 = ret_xy11['probs']
    prob_xy22 = ret_xy22['probs']
    prob_yx11 = ret_yx11['probs']
    prob_yx22 = ret_yx22['probs']

    def compute_cost(prob, y, y_mask, att, t, t_mask, g):
        _y = tensor.eq(y, 1)
        y_mask *= ((1 - _y) + _y * (1 - t_mask))

        # normal loss
        ccost = -tensor.log(compute_prob(prob, y, y_mask) * g +
                            compute_prob(att, t, t_mask) * (1 - g) +
                            1e-7)
        ccost = (ccost * (1 - (1 - y_mask) * (1 - t_mask))).sum(0)

        # gate loss
        gcost = -(tensor.log(g) * (1 - t_mask) + tensor.log(1-g) * t_mask)
        gcost = (gcost * (1 - (1 - y_mask) * (1 - t_mask))).sum(0)

        return ccost, gcost

    # get cost
    cost_xy1, g_cost_xy1 = compute_cost(prob_xy11, y1, y1_mask, att_xy12, txy12, txy12_mask, gate_xy1)
    cost_xy2, g_cost_xy2 = compute_cost(prob_xy22, y2, y2_mask, att_xy21, txy21, txy21_mask, gate_xy2)
    cost_yx1, g_cost_yx1 = compute_cost(prob_yx11, x1, x1_mask, att_yx12, tyx12, tyx12_mask, gate_yx1)
    cost_yx2, g_cost_yx2 = compute_cost(prob_yx22, x2, x2_mask, att_yx21, tyx21, tyx21_mask, gate_yx2)
    cost   = cost_xy1 + cost_xy2 + cost_yx1 + cost_yx2
    g_cost = g_cost_xy1 + g_cost_xy2 + g_cost_yx1 + g_cost_yx2

    print 'build sampler (one-step)'
    f_init_xy, f_next_xy = build_sampler(tparams_xy, options, options['trng'], 'xy_')
    f_init_yx, f_next_yx = build_sampler(tparams_yx, options, options['trng'], 'yx_')

    print 'build old sampler'
    f_init_xy0, f_next_xy0 = build_sampler(tparams_xy0, options, options['trng'], 'xy_')
    f_init_yx0, f_next_yx0 = build_sampler(tparams_yx0, options, options['trng'], 'yx_')

    print 'build attender (one-step)'
    f_attend_xy = build_attender(tparams_xy, None, options, 'xy_', one_step=True)  # E->F curr
    f_attend_yx = build_attender(tparams_yx, None, options, 'yx_', one_step=True)

    # before any regularizer
    print 'build Cost Function...',
    inputs = [x1, x1_mask, y1, y1_mask, x2, x2_mask, y2, y2_mask,
              txy12, txy12_mask, txy21, txy21_mask,
              tyx12, tyx12_mask, tyx21, tyx21_mask]
    f_valid = theano.function(inputs, cost, profile=profile)

    print 'build Gradient (backward)...',
    if options['build_gate']:
        tparams = dict(tparams_xy.items() + tparams_yx.items() + tparams_gate.items())
    else:
        tparams = dict(tparams_xy.items() + tparams_yx.items())

    cost   = cost.mean()
    g_cost = g_cost.mean()

    if options['gate_loss']:
        grads = clip(tensor.grad(cost + options['gate_lambda'] * g_cost,
                                 wrt=itemlist(tparams)), options['clip_c'])
    else:
        grads = clip(tensor.grad(cost, wrt=itemlist(tparams)),
                     options['clip_c'])
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    outputs = [cost, g_cost]
    print 'Building Optimizers...',
    f_cost, f_update = eval(options['optimizer'])(
        lr, tparams, grads, inputs, outputs)

    print 'Done'

    # put everything into function lists
    funcs['valid']    = f_valid
    funcs['cost']     = f_cost
    funcs['update']   = f_update

    funcs['init_xy']  = f_init_xy
    funcs['init_yx']  = f_init_yx
    funcs['next_xy']  = f_next_xy
    funcs['next_yx']  = f_next_yx

    funcs['init_xy0'] = f_init_xy0
    funcs['init_yx0'] = f_init_yx0
    funcs['next_xy0'] = f_next_xy0
    funcs['next_yx0'] = f_next_yx0

    funcs['att_xy']   = f_attend_xy
    funcs['att_yx']   = f_attend_yx

    funcs['crit_xy'] = ret_xy11['f_critic']
    funcs['crit_yx'] = ret_yx11['f_critic']

    if options['build_gate']:
        funcs['gate']    = f_gate

    print 'Build Networks... done!'
    return funcs, [tparams, tparams_xy0, tparams_yx0]

funcs, tp = build_networks(model_options)
tparams, tparams_xy0, tparams_yx0 = tp

# print 'save the compiled functions/tparams for temperal usage'


print 'Loading data'
train = TextIterator(model_options['datasets'], model_options['dictionaries'], [0, 0, 0, 0],
                     batch_size=model_options['batch_size'], maxlen=model_options['maxlen'])
valid = TextIterator(model_options['valid_datasets'], model_options['dictionaries'], [0, 0, 0, 0],
                     batch_size=model_options['batch_size'], maxlen=200)


print clr('-------------------------------------------- Main-Loop -------------------------------------------------',
          'yellow')

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
def idx2seq(x, ii, pp=None):
    seq = []
    for kk, vv in enumerate(x):
        if vv == 0:
            break
        if vv in worddicts_r[ii]:
            word = worddicts_r[ii][vv]

            if pp is None:
                if vv > model_options['voc_sizes'][ii]:
                    seq.append(clr(word, 'green'))
                else:
                    seq.append(word)
            else:
                if pp[kk] == 0:
                    seq.append(clr(word, 'red'))
                elif (pp[kk] > 0) and (pp[kk] <= 0.25):
                    seq.append(clr(word, 'yellow'))
                elif (pp[kk] > 0.25) and (pp[kk] <= 0.5):
                    seq.append(clr(word, 'green'))
                elif (pp[kk] > 0.5) and (pp[kk] <= 0.75):
                    seq.append(clr(word, 'cyan'))
                else:
                    seq.append(clr(word, 'blue'))

        else:
            seq.append(clr('UNK', 'white'))
    return ' '.join(seq)


# compute-update
@Timeit
def execute(inps, lrate, info):
    eidx, uidx = info
    cost, g_cost = funcs['cost'](*inps)
    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'G', g_cost

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

        # print 'x1:{}, x2:{}, y1:{}, y2:{}'.format(x1.shape, x2.shape, y1.shape, y2.shape)

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
        ty21, ty21_mask = prepare_cross(sy2, sy1, y2.shape[0])

        # print 'x1:{}, x2:{}, y1:{}, y2:{}'.format(x1.shape, x2.shape, y1.shape, y2.shape)

        inps = [x1, x1_mask, y1, y1_mask,
                x2, x2_mask, y2, y2_mask,
                ty12, ty12_mask, ty21, ty21_mask,
                tx12, tx12_mask, tx21, tx21_mask]

        try:
            execute(inps, lrate, [eidx, uidx])  # train one step.

        except Exception, e:
            print clr(e, 'red')
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
                sample, sc, acts, gg = gen_sample_memory(tparams, funcs,
                                                         x1[:, jj][:, None],
                                                         x2[:, jj][:, None],
                                                         y2[:, jj][:, None],
                                                         model_options,
                                                         rng=model_options['rng'],
                                                         m=0, k=1, maxlen=200,
                                                         stochastic=model_options['stochastic'],
                                                         argmax=True)

                sample0, sc0  = gen_sample(tparams_xy0,
                                           funcs['init_xy0'],
                                           funcs['next_xy0'],
                                           x1[:, jj][:, None],
                                           model_options,
                                           rng=model_options['rng'], k=1,
                                           maxlen=200,
                                           stochastic=model_options['stochastic'],
                                           argmax=True)

                print '============================='
                print 'Target-TM {}: {}'.format(jj, idx2seq(sy2[jj], 3))
                print 'Source-TM {}: {}'.format(jj, idx2seq(sx2[jj], 2))
                print 'Source-CR {}: {}'.format(jj, idx2seq(sx1[jj], 0))
                print 'Target-CR {}: {}'.format(jj, idx2seq(sy1[jj], 1))
                print '-----------------------------'

                if model_options['stochastic']:
                    ss = sample
                else:
                    sc /= numpy.array([len(s) for s in sample])
                    ss = sample[sc.argmin()]
                ss0 = sample0

                _ss = []
                for ii, si in enumerate(ss):
                    if si < model_options['voc_sizes'][1]:
                        _ss.append(si)
                    else:
                        offset = si - model_options['voc_sizes'][1]
                        _ss.append(sy2[jj][offset])

                # print 'Sample-CR {}: {}'.format(jj, idx2seq(_ss, 1))
                print 'NMT Model {}: {}'.format(jj, idx2seq(ss0, 1))
                print 'Copy Prob {}: {}'.format(jj, idx2seq(_ss, 1, acts))
                print 'Copy Gate {}: {}'.format(jj, idx2seq(_ss, 1, gg))
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



