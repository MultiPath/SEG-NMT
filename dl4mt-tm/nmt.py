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
        attention_prop = opt_ret['attention'] * y_mask[:, :, None]
        f_critic = theano.function(inps, [attention_prop, opt_ret['logit']],
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
def gen_sample_memory(tparams,
               funcs,
               x1, x2, y2,
               options,
               rng=None,
               m=0,
               k=1,  # beam-size
               maxlen=200,
               stochastic=True, argmax=False):
    # modes
    modes = ['xy', 'yx']
    l_max = options['voc_sizes'][1]

    # masks
    x1_mask = numpy.array(x1 > 0, dtype='float32')
    x2_mask = numpy.array(x2 > 0, dtype='float32')
    y2_mask = numpy.array(y2 > 0, dtype='float32')

    # k is the beam size we have
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    action = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_actions = [[]] * live_k
    hyp_scores  = numpy.zeros(live_k).astype('float32')
    hyp_states  = []

    # get initial state of decoder rnn and encoder context for x1
    ret = funcs['init_' + modes[m]](x1)
    next_state, ctx0 = ret[0], ret[1]  # init-state, contexts
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # get translation memory encoder context
    _, mctx0 = funcs['init_' + modes[m]](x2)

    # get attention propagation for translation memory
    attpipe, _ = funcs['crit_' + modes[1 - m]](y2, y2_mask, x2, x2_mask)
    attpipe = numpy.squeeze(attpipe)

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        mctx = numpy.tile(mctx0, [live_k, 1])

        # -- mask OOV words as UNK
        _next_w = (next_w * (next_w < l_max) + 1.0 * (next_w >= l_max)).astype('int64')

        # --copy mode
        ret = funcs['att_' + modes[m]](next_state, _next_w, mctx)
        mctxs, matt, mattsum = ret[0], ret[1], ret[2]    # matt: batchsize x len_x2
        copy_p = numpy.dot(matt, attpipe)  # batchsize x len_y2

        # --generate mode
        ret = funcs['next_' + modes[m]](_next_w, ctx, next_state)
        next_p, next_w, next_state, ctxs, attsum = ret[0], ret[1], ret[2], ret[3], ret[4]

        # compute gate
        if not options['build_gate']:
            gates = numpy.clip(mattsum / (attsum + mattsum), 0, 1) # Natural Gate.
        else:
            gates = funcs['gate'](
                next_state[None, :, :],
                ctxs[None, :, :],
                mctxs[None, :, :])[0]  # batchsize

        # real probabilities
        next_p *= (1 - gates[:, None])
        copy_p *= gates[:, None]

        def _merge():
            temp_p = copy.copy(numpy.concatenate([next_p, copy_p], axis=1))

            for i in range(next_p.shape[0]):
                for j in range(copy_p.shape[1]):
                    if y2[j] != 1:
                        temp_p[i, y2[j]] += copy_p[i, j]
                        temp_p[i, l_max + j] = 0.
                temp_p[i, 1] = 0. # never output UNK
            # temp_p -= 1e-8
            return temp_p

        merge_p = _merge()

        if stochastic:
            if argmax:
                nw = merge_p[0].argmax()
                next_w[0] = nw
            else:
                nw = rng.multinomial(1, pvals=merge_p[0]).argmax()

            sample.append(nw)
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
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
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

    return sample, sample_score, action

