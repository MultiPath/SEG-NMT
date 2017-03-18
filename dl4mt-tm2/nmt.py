'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
import copy
from layer import *
from optimizer import *


# initialize all parameters
def init_params(options, pix=''):
    params = OrderedDict()

    # embedding
    params[pix+'Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params[pix+'Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    # ==> we have two encoder-decoder in one model

    params = get_layer(options['encoder'])[0](options, params,
                                              prefix=pix+'encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix=pix+'encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix=pix+'ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix=pix+'decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix=pix+'ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix=pix+'ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix=pix+'ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix=pix+'ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, inps, options, pix='', return_cost=False, with_compile=False):
    opt_ret = dict()

    # deal with the input
    x, x_mask, y, y_mask = inps

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams[pix+'Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=pix+'encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams[pix+'Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix=pix+'encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # save the contexts
    opt_ret['ctx'] = ctx

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix=pix+'ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams[pix+'Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix=pix+'decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['prev_hids'] = concatenate([init_state[None, :, :], proj_h[:-1, :, :]], axis=0)
    opt_ret['prev_emb']  = emb
    opt_ret['ctxs'] = ctxs
    opt_ret['attention'] = proj[2]
    opt_ret['att_sum'] = proj[3]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix=pix+'ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix=pix+'ff_logit_prev', activ='linear')
    logit_ctx  = get_layer('ff')[1](tparams, ctxs, options,
                                    prefix=pix+'ff_logit_ctx',  activ='linear')

    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix=pix+'ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    opt_ret['hids']   = proj_h
    opt_ret['probs']  = probs
    opt_ret['logit']  = logit

    # cost
    if return_cost:
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
        cost = -tensor.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)
        opt_ret['cost'] =  cost

    if with_compile:
        print 'Build f_critic...',
        f_critic = theano.function(inps, [opt_ret['hids'], opt_ret['ctxs'], opt_ret['logit']],
                                   name='f_critic', profile=profile)

        opt_ret['f_critic'] = f_critic
        print 'Done'

    return opt_ret


# build an attender
# build a training model
def build_attender(tparams, inps, options, pix='', one_step=False):
    opt_ret = dict()

    # deal with the input
    if not one_step:
        prev_hids, prev_emb, ctx, x_mask = inps
    else:
        prev_hids = tensor.matrix('onestep_p_hs', dtype='float32')
        prev_word = tensor.vector('onestep_p_w',  dtype='int64')
        prev_emb  = tensor.switch(prev_word[:, None] < 0,
                    tensor.alloc(0., 1, tparams[pix+'Wemb_dec'].shape[1]),
                    tparams[pix+'Wemb_dec'][prev_word])

        ctx       = tensor.tensor3('onestep_ctx', dtype='float32')
        x_mask    = None
        inps = [prev_hids, prev_word, ctx]

    def recurrence(hid, emb, ctx, x_mask):
        proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                                prefix=pix+'decoder',
                                                context=ctx,
                                                context_mask=x_mask,
                                                one_step=True,
                                                init_state=hid)
        ctxs = proj[1]
        atts = proj[2]
        att_sum = proj[3]
        return ctxs, atts, att_sum

    if not one_step:
        ret, _ = theano.scan(recurrence, sequences=[prev_hids, prev_emb],
                             non_sequences=[ctx, x_mask])
        # weights (alignment matrix)
        opt_ret['ctxs'] = ret[0]
        opt_ret['attention'] = ret[1]
        opt_ret['att_sum'] = ret[2]
        return opt_ret

    else:
        ret = recurrence(prev_hids, prev_emb, ctx, x_mask)
        print 'Build f_attend...',

        f_attend = theano.function(inps, ret, name='f_attend', profile=profile)
        print 'Done.'

        return f_attend


# build a sampler
# build a sampling model
def build_sampler(tparams, options, trng, pix=''):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb  = tparams[pix+'Wemb'][x.flatten()]
    emb  = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams[pix+'Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=pix+'encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix=pix+'encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix=pix+'ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams[pix+'Wemb_dec'].shape[1]),
                        tparams[pix+'Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix=pix+'decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]
    atts = proj[3]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix=pix+'ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix=pix+'ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix=pix+'ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix=pix+'ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next...',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state, ctxs, atts]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams,
               f_init, f_next,
               x,
               options,
               rng=None,
               k=1, maxlen=200,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample_memory(tparams, funcs,
                      x1, x2, y2,
                      options, rng=None,
                      m=0, k=1, maxlen=200,
                      stochastic=True, argmax=False):
    l_max = options['voc_sizes'][1-m]

    # masks
    x1_mask = numpy.array(x1 > 0, dtype='float32')
    x2_mask = numpy.array(x2 > 0, dtype='float32')
    y2_mask = numpy.array(y2 > 0, dtype='float32')
    x1_mask[1:] = x1_mask[:-1]
    x2_mask[1:] = x2_mask[:-1]
    y2_mask[1:] = y2_mask[:-1]

    # k is the beam size we have
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    action = []
    gating = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_actions = [[]] * live_k
    hyp_gatings = [[]] * live_k
    hyp_scores  = numpy.zeros(live_k).astype('float32')

    # get initial state of decoder rnn and encoder context for x1
    ret = funcs['init_xy'](x1)
    next_state, ctx0 = ret[0], ret[1]  # init-state, contexts
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # get attention propagation for translation memory
    hids20, ctxs20, _ = funcs['crit_xy'](x2, x2_mask, y2, y2_mask)

    # initial coverage vector
    if options.get('nn_coverage', False):
        next_cov = numpy.zeros((live_k, y2.shape[0], options['cov_dim']), dtype='float32')
    else:
        next_cov = numpy.zeros((live_k, y2.shape[0]), dtype='float32')

    for ii in xrange(maxlen):
        ctx   = numpy.tile(ctx0,   [live_k, 1])
        ctxs2 = numpy.tile(ctxs20, [live_k, 1])
        hids2 = numpy.tile(hids20, [live_k, 1])
        y2_mask_ = numpy.tile(y2_mask, [live_k])

        # -- mask OOV words as UNK
        _next_w = (next_w * (next_w < l_max) + 1.0 * (next_w >= l_max)).astype('int64')

        # --generate mode
        ret = funcs['next_xy'](_next_w, ctx, next_state)
        next_p, next_w, next_state, ctxs, attsum = ret[0], ret[1], ret[2], ret[3], ret[4]

        # -- compute mapping, gating and copying attention
        if not options['use_coverage']:
            outs = funcs['map'](ctxs[None, :, :], ctxs2,
                                next_state[None, :, :], hids2,
                                y2_mask_)
            mapping, gates, copy_p = [o[0] for o in outs]

        else:
            outs = funcs['map'](ctxs[None, :, :], ctxs2,
                                next_state[None, :, :], hids2,
                                y2_mask_, next_cov)
            mapping, gates, copy_p, next_cov = [o[0] for o in outs]

        # real probabilities
        next_p *= 1 - gates[:, None]
        copy_p *= gates[:, None]

        def _merge():
            temp_p = copy.copy(numpy.concatenate([next_p, copy_p], axis=1))

            for i in range(next_p.shape[0]):
                for j in range(copy_p.shape[1]):
                    if y2[j] != 1:
                        temp_p[i, y2[j]] += copy_p[i, j]
                        temp_p[i, l_max + j] = 0.
                temp_p[i, 1] = 0.  # never output UNK
            return temp_p

        merge_p = _merge()

        if stochastic:
            if argmax:
                nw = merge_p[0].argmax()
                next_w[0] = nw
            else:
                nw = rng.multinomial(1, pvals=merge_p[0]).argmax()

            sample.append(nw)
            gating.append(1 - gates[:, None])
            if nw >= l_max:
                action.append(0.0)
            else:
                action.append(next_p[0, nw] / merge_p[0, nw])

            # action.append(gates[0])
            sample_score -= numpy.log(merge_p[0, nw])
            if nw == 0:
                break
        else:
            # TODO: beam-search is still not ready.
            cand_scores = hyp_scores[:, None] - numpy.log(merge_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = merge_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores  = numpy.zeros(k - dead_k).astype('float32')
            new_hyp_states  = []
            new_hyp_covs    = []
            new_hyp_gatings = []
            new_hyp_actions = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_covs.append(copy.copy(next_cov[ti]))
                new_hyp_gatings.append(hyp_gatings[ti] + [1 - gates[ti]])
                if wi >= l_max:
                    new_hyp_actions.append(hyp_actions[ti] + [0])
                else:
                    new_hyp_actions.append(hyp_actions[ti] + [next_p[0, wi] / merge_p[0, wi]])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores  = []
            hyp_states  = []
            hyp_covs    = []
            hyp_gatings = []
            hyp_actions = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    action.append(new_hyp_actions[idx])
                    gating.append(new_hyp_gatings[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_covs.append(new_hyp_covs[idx])
                    hyp_gatings.append(new_hyp_gatings[idx])
                    hyp_actions.append(new_hyp_actions[idx])

            hyp_scores = numpy.array(hyp_scores)
#             hyp_gating = numpy.array(hyp_gating)
#             hyp_action = numpy.array(hyp_action)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_cov   = numpy.array(hyp_covs)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                action.append(hyp_actions[idx])
                gating.append(hyp_gatings[idx])

    return sample, sample_score, action, gating


@Timeit
def build_networks(options, model=' ', train=True):
    funcs = dict()

    print 'Building model: X -> Y & Y -> X model'
    params_xy  = init_params(options, 'xy_')
    params_xy0 = copy.copy(params_xy)
    print 'Done.'

    print 'load the pretrained NMT-models...'
    params_xy0  = load_params2(options['baseline_xy'], params_xy0, mode='xy_')
    tparams_xy0 = init_tparams(params_xy0)  # pre-trained E->F model
    print 'Done.'

    # use pre-trained models
    if options['use_pretrain']:
        params_xy = copy.copy(params_xy0)

    # reload parameters
    if train:
        if options['reload_'] and os.path.exists(options['saveto']):
            print 'Reloading pre-saved parameters'
            params_xy = load_params(options['saveto'], params_xy)
        else:
            print 'Start a new model'
    else:

        print 'Reloading NMT parameters'
        params_xy = load_params(model, params_xy)

    tparams_xy = init_tparams(params_xy)

    # inputs of the model (x1, y1, x2, y2)
    x1 = tensor.matrix('x1', dtype='int64')
    x1_mask = tensor.matrix('x1_mask', dtype='float32')
    y1 = tensor.matrix('y1', dtype='int64')
    y1_mask = tensor.matrix('y1_mask', dtype='float32')
    x2 = tensor.matrix('x2', dtype='int64')
    x2_mask = tensor.matrix('x2_mask', dtype='float32')
    y2 = tensor.matrix('y2', dtype='int64')
    y2_mask = tensor.matrix('y2_mask', dtype='float32')

    # TM-rxyerence index
    txy12 = tensor.matrix('xy12', dtype='int64')
    txy12_mask = tensor.matrix('xy12_mask', dtype='float32')

    print 'build forward-attention models (2 models simultaneously)...'
    ret_xy11 = build_model(tparams_xy, [x1, x1_mask, y1, y1_mask], options, 'xy_', False, True)   # X->Y curr
    ret_xy22 = build_model(tparams_xy, [x2, x2_mask, y2, y2_mask], options, 'xy_', False, False)  # X->Y tm

    print 'build mapping (bi-linear model)!'
    params_map  = OrderedDict()

    # use diagonal matrix
    if options['diagonal']:
        print 'use diagonal matrix'

        bmap = 'bd'
    else:
        print 'use full matrix'

        bmap = 'bi'

    # params for copying
    if not options['use_coverage']:
        params_map  = get_layer(bmap)[0](options, params_map, prefix='map_bi',
                                         nin1=2 * options['dim'],
                                         nin2=2 * options['dim'], eye=True)
    else:
        if not options.get('nn_coverage', False):
            print 'use linguistic coverage'
            params_map = get_layer(bmap)[0](options, params_map, prefix='map_bi',
                                            nin1=2 * options['dim'],
                                            nin2=2 * options['dim'],
                                            bias=True, eye=True)
        else:
            print 'use neural network coverage'
            params_map = get_layer('bg')[0](options, params_map, prefix='map_bg',
                                            nin1=2 * options['dim'],
                                            nin2=2 * options['dim'],
                                            bias=True, eye=True)
            params_map = get_layer('gru')[0](options, params_map,
                                             prefix='gru_map',
                                             nin=4 * options['dim'] + 1,
                                             dim=options['cov_dim'])

    params_map['tau'] = numpy.float32(0.5)  # 1.0   # temperature for copy

    # params for gating
    params_map = get_layer('ff')[0](options, params_map, prefix='map_ff',
                                    nin=4 * options['dim'], nout=2)

    # reload parameters
    if train:
        if options['reload_'] and os.path.exists(options['saveto']):
            print 'Reloading pre-saved parameters'
            params_map = load_params(options['saveto'], params_map)
            print 'Done.'

    else:
        print 'Reloading mapping parameters'
        params_map = load_params(model, params_map)
        print 'Done.'

    tparams_map = init_tparams(params_map)

    if not options.get('nn_coverage', False):
        att0 = tensor.matrix('init_atten')
    else:
        att0 = tensor.tensor3('init_atten')

    inps = []
    outs = []

    if not options['use_coverage']:

        inps += [ret_xy11['ctxs'], ret_xy22['ctxs'], ret_xy11['hids'], ret_xy22['hids'], y2_mask]

        def build_mapping(ctx1, ctx2):
            # ctx1 = normalize(ctx1)
            # ctx2 = normalize(ctx2)
            return get_layer(bmap)[1](tparams_map, ctx1, ctx2,
                                      prefix='map_bi', activ='lambda x: x')

        mapping = build_mapping(inps[0], inps[1])

        # copy: alternative
        tm_mask = y2_mask.T
        attens  = softmax(mapping * tparams_map['tau'], mask=tm_mask[None, :, :])

        # gate: alternative
        # gates   = sigmoid(tensor.max(mapping, axis=-1) * tparams_map['eta'])
        att_tmh = tensor.batched_dot(attens.dimshuffle(1, 0, 2),            # dec_cur x bs x dec_tm
                                     ret_xy22['hids'].dimshuffle(1, 0, 2))  # dec_tm x bs x hid_dim
                                                                            # bs x dec_cur x hid_dim
        att_tmh = att_tmh.dimshuffle(1, 0, 2)

        gates = get_layer('ff')[1](tparams_map,
                                   concatenate([ret_xy11['hids'], att_tmh, ret_xy11['ctxs']], axis=2),
                                   options, prefix='map_ff', activ='softmax')[:, :, 0]

        outs += [mapping, gates, attens]

    else:

        inps += [ret_xy11['ctxs'], ret_xy22['ctxs'],
                 ret_xy11['hids'], ret_xy22['hids'],
                 y2_mask, att0]


        # cur_ctx1: batch_size x context_dim
        # tm_ctx2:  dec_tm x batch_size x context_dim
        # prev_att: batch_size x dec_tm
        # map : dec_cur x batch_size x dec_tm
        def build_mapping_step(cur_ctx1, cur_hid,
                               prev_att,
                               tm_ctx2, tm_hids, tm_mask):
            # normalize
            # cur_ctx1_ = normalize(cur_ctx1)
            if not options.get('nn_coverage', False):
                mapping   = get_layer(bmap)[1](tparams_map, cur_ctx1[None, :, :],
                                              tm_ctx2, prev_att[None, :, :],
                                              prefix='map_bi', activ='lambda x: x')[0]  # batchsize x dec_tm
                attens    = softmax(mapping * tparams_map['tau'], mask=tm_mask)
                coverage  = prev_att + attens

                att_tmh   = tensor.batched_dot(attens[:, None, :],            # bs x dec_tm
                                           tm_hids.dimshuffle(1, 0, 2))    # dec_tm x bs x hid_dim
                                                                           # bs x 1 x hid_dim
                att_tmh   = att_tmh[:, 0, :]                                   # bs x hid_dim

                gates     = get_layer('ff')[1](tparams_map,
                                               concatenate([cur_hid, att_tmh, cur_ctx1], axis=1),
                                               options, prefix='map_ff', activ='softmax')[:, 0]
            else:

                mapping  = get_layer('bg')[1](tparams_map, cur_ctx1[None, :, :],
                                              tm_ctx2, prev_att,
                                              prefix='map_bg', activ='lambda x: x')[0]  # # 1 x bs x dec_tm
                attens   = softmax(mapping * tparams_map['tau'], mask=tm_mask)

                tm_ctx2_shape = tm_ctx2.shape
                tm_ctx2_  = tm_ctx2.reshape((tm_ctx2_shape[0] * tm_ctx2_shape[1], tm_ctx2_shape[2]))

                cur_ctx1  = cur_ctx1[None, :, :]
                cur_ctx1_ = tensor.repeat(cur_ctx1, tm_ctx2_shape[0], axis=0).reshape(
                                (tm_ctx2_shape[0] * tm_ctx2_shape[1], tm_ctx2_shape[2]))

                attens_   = attens.dimshuffle(1, 0)
                attens_   = attens_[:, :, None]
                attens_   = attens_.reshape((tm_ctx2_shape[0] * tm_ctx2_shape[1], 1))

                state_below = concatenate([tm_ctx2_, cur_ctx1_, attens_], axis=1)
                state_below = state_below[None, :, :]

                coverage    = get_layer('gru')[1](tparams_map, state_below, options, prefix='gru_map')
                coverage    = coverage[0].reshape((tm_ctx2_shape[1], tm_ctx2_shape[0], options['cov_dim']))
                att_tmh     = tensor.batched_dot(attens[:, None, :],  # bs x dec_tm
                                                 tm_hids.dimshuffle(1, 0, 2))  # dec_tm x bs x hid_dim
                # bs x 1 x hid_dim
                att_tmh     = att_tmh[:, 0, :]  # bs x hid_dim

                gates       = get_layer('ff')[1](tparams_map,
                                                 concatenate([cur_hid, att_tmh, cur_ctx1], axis=1),
                                                 options, prefix='map_ff', activ='softmax')[:, 0]

            return coverage, attens, mapping, gates

        ret, _ = theano.scan(build_mapping_step,
                             sequences=[ret_xy11['ctxs'], ret_xy11['hids']],
                             outputs_info=[att0, None, None, None],
                             non_sequences=[ret_xy22['ctxs'], ret_xy22['hids'], y2_mask.T])

        coverage, attens, mapping, gates = ret[0], ret[1], ret[2], ret[3]

        outs += [mapping, gates, attens, coverage]


    print 'Building Mapping functions, ...',
    f_map = theano.function(inps, outs, profile=profile)
    print 'Done.'

    print 'build loss function (w/o gate)'

    # get the loss function
    def compute_prob(probs, y, y_mask=None):

        # compute the loss for the vocabulary-selection side
        y_flat  = y.flatten()
        n_words = probs.shape[-1]
        y_flat_idx = tensor.arange(y_flat.shape[0]) * n_words + y_flat
        probw   = probs.flatten()[y_flat_idx]
        probw   = probw.reshape([y.shape[0], y.shape[1]])
        if y_mask is not None:
           probw  *= y_mask
        return probw

    def compute_cost(prob, y, y_mask, att, t, t_mask, g):
        _y = tensor.eq(y, 1)
        y_mask *= ((1 - _y) + _y * (1 - t_mask))

        # normal loss ---> maybe neumerically unstable
        ccost = -tensor.log(tensor.clip(
                            compute_prob(prob, y, y_mask) * (1 - g) +
                            compute_prob(att, t, t_mask) * g, 1e-7, 1 - 1e-7))
        ccost = (ccost * (1 - (1 - y_mask) * (1 - t_mask))).sum(0)

        # alternative loss
        # ccost = -tensor.log(compute_prob(prob, y)) * (1 - g) * y_mask \
        #         -tensor.log(compute_prob(att, t))  * g       * t_mask
        # ccost = ccost.sum(0)

        # gate loss
        gcost = -(tensor.log(1 - g) * (1 - t_mask) + tensor.log(g) * t_mask)
        gcost = (gcost * (1 - (1 - y_mask) * (1 - t_mask))).sum(0)

        return ccost, gcost

    probs = ret_xy11['probs']
    cost, g_cost = compute_cost(probs, y1, y1_mask, attens, txy12, txy12_mask, gates)

    print 'build sampler (one-step)'
    f_init_xy, f_next_xy   = build_sampler(tparams_xy, options, options['trng'], 'xy_')

    print 'build old sampler'
    f_init_xy0, f_next_xy0 = build_sampler(tparams_xy0, options, options['trng'], 'xy_')

    if train:
        # before any regularizer
        print 'build Cost Function...',
        inputs = [x1, x1_mask, y1, y1_mask,
                  x2, x2_mask, y2, y2_mask,
                  txy12, txy12_mask]
        if options['use_coverage']:
            inputs += [att0]

        f_valid = theano.function(inputs, cost, profile=profile)

        print 'build Gradient...',
        tparams = dict(tparams_xy.items() + tparams_map.items())

        cost   = cost.mean()
        g_cost = g_cost.mean()

        if options['only_train_g']:
            _tparams = tparams_map
        else:
            _tparams = tparams

        if options['gate_loss']:
            grads, g2 = clip(tensor.grad(cost + options['gate_lambda'] * g_cost,
                                     wrt=itemlist(_tparams)), options['clip_c'])
        else:
            grads0 = tensor.grad(cost, wrt=itemlist(_tparams))
            grads, g2 = clip(grads0, options['clip_c'])
        print 'Done'

        # compile the optimizer, the actual computational graph is compiled here
        lr = tensor.scalar(name='lr')
        outputs = [cost, g_cost, tensor.sqrt(g2)]


        print 'Building Optimizers...',
        f_cost, f_update = eval(options['optimizer'])(
            lr, _tparams, grads, inputs, outputs)

        funcs['valid']  = f_valid
        funcs['cost']   = f_cost
        funcs['update'] = f_update

    else:
        tparams = dict(tparams_xy.items() + tparams_map.items())
    print 'Done'

    # put everything into function lists
    funcs['init_xy']   = f_init_xy
    funcs['next_xy']   = f_next_xy
    funcs['crit_xy']   = ret_xy11['f_critic']
    funcs['init_xy0']  = f_init_xy0
    funcs['next_xy0']  = f_next_xy0
    funcs['map']       = f_map

    print 'Build Networks... done!'
    if train:
        return funcs, [tparams, tparams_xy0]
    else:
        return funcs, tparams
