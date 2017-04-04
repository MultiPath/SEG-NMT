'''
Building blocks of  Neural Machine Translation
'''

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

try:
    from pycrayon import CrayonClient
except ImportError:
    pass

from collections import OrderedDict


profile = False

class flushfile(object):
    def __getattr__(self,name):
        return object.__getattribute__(self.f, name)
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

import sys
sys.stdout = flushfile(sys.stdout)




class Timeit(object):
    def __init__(self, func):
        self._wrapped = func

    def __call__(self, *args, **kws):
        start_t = time.time()
        result = self._wrapped(*args, **kws)
        print '{}: elapsed {:.4f} secs.'.format(self._wrapped.__name__,
                                                time.time() - start_t)
        return result


class Monitor(object):
    def __init__(self, address, port):
        self.cc  = CrayonClient(hostname=address, port=port)

    def start_experiment(self, name):
        exps = self.cc.get_experiment_names()
        if name in exps:
            self.exp = self.cc.open_experiment(name)

        else:
            self.exp = self.cc.create_experiment(name)

    def push(self, data, wall_time=-1, step=-1):
        self.exp.add_scalar_dict(data, wall_time, step)


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    print 'loading ...'
    for it, (kk, vv) in enumerate(params.iteritems()):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        print '{}\t'.format(kk),
        if it % 5 == 0:
            print ''
        params[kk] = pp[kk]

    return params


# load parameters-2
def load_params2(path, params, mode=''):
    pp = numpy.load(path)
    print 'loading ...'
    for it, (kk, vv) in enumerate(params.iteritems()):
        if kk[:3] == mode:
            if kk[3:] not in pp:
                warnings.warn('%s is not in the archive' % kk)
                continue

            print kk,
            if it % 5 == 0:
                print ''
            params[kk] = pp[kk[3:]]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'bi': ('param_init_bllayer', 'bllayer'),
          'bd': ('param_init_bdlayer', 'bdlayer'),
          'bg': ('param_init_bglayer', 'bglayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def sigmoid(x):
    return tensor.nnet.sigmoid(x)


def softmax(x, mask=None):
    if not mask:
        if x.ndim == 2:
            return tensor.nnet.softmax(x)
        else:
            shp  = x.shape
            prob = tensor.nnet.softmax(
                   x.reshape((shp[0] * shp[1], shp[2])))
            return prob.reshape(x.shape)
    else:
        max_x = x.max(axis=-1, keepdims=True)
        exp_x = tensor.exp(x - max_x) * mask
        prob  = exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-8)
        return prob


def linear(x):
    return x


def normalize(x):
    x_norm = tensor.sqrt(tensor.sum(x ** 2, axis=-1, keepdims=True))
    return x / x_norm


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# bi-linear layer:
def param_init_bllayer(options, params, prefix='bi',
                       nin1=None, nin2=None, eye=False,
                       bias=False):
    if not nin2:
        nin2 = nin1

    if not eye:
        params[_p(prefix, 'M')] = norm_weight(nin1, nin2, scale=0.01, ortho=True)
    else:
        print 'bi-eye init'
        params[_p(prefix, 'M')] = numpy.eye(nin1, nin2, dtype='float32')

    if bias:
        params[_p(prefix, 'b')] = numpy.float32(0.)

    return params


def bllayer(tparams, input1, input2, cov=None, prefix='bi',
            activ='lambda x: tensor.nnet.sigmoid(x)',
            **kwargs):

    # if cov is not None:
    #     assert (_p(prefix, 'b') in tparams, 'coverage as bias')

    input1 = tensor.dot(input1, tparams[_p(prefix, 'M')])
    if input1.ndim == 2:
        output = tensor.dot(input1, input2.dimshuffle(1, 0))
    else:
        output = tensor.batched_dot(input1.dimshuffle(1, 0, 2),
                                    input2.dimshuffle(1, 2, 0))
        output = output.dimshuffle(1, 0, 2)  # dec_len x batch_size x enc_len

    if cov is not None:
        output += tparams[_p(prefix, 'b')] * cov
    # output = output.reshape((output.shape[1], output.shape[2]))
    return eval(activ)(output)

def param_init_bglayer(options, params, prefix='bg',
                       nin1=None, nin2=None, eye=False,
                       bias=False):
    if not nin2:
        nin2 = nin1

    if not eye:
        params[_p(prefix, 'M')] = norm_weight(nin1, nin2, scale=0.01, ortho=True)
    else:
        print 'bg-eye init'
        params[_p(prefix, 'M')] = numpy.eye(nin1, nin2, dtype='float32')

    if bias:
        params[_p(prefix, 'b')] = numpy.zeros((10,), dtype = 'float32')

    return params


def bglayer(tparams, input1, input2, cov=None, prefix='bg',
            activ='lambda x: tensor.nnet.sigmoid(x)',
            **kwargs):

    # if cov is not None:
    #     assert (_p(prefix, 'b') in tparams, 'coverage as bias')

    input1 = tensor.dot(input1, tparams[_p(prefix, 'M')])     #1 x batch_size x c_dim
    if input1.ndim == 2:
        output = tensor.dot(input1, input2.dimshuffle(1, 0))
    else:
        output = tensor.batched_dot(input1.dimshuffle(1, 0, 2),   #bs x 1 x c_dim
                                    input2.dimshuffle(1, 2, 0))    # bs x c_dim x dec_tm
        output = output.dimshuffle(1, 0, 2)  # 1 x batch_size x dec_tm

    if cov:    #bs x dec_tm x d
        output += (cov * tparams[_p(prefix, 'b')][None, None, :]).sum(-1)[None, :, :]  # 1 x bs x dec_tm

    return eval(activ)(output)


# bi-linear layer with diagonal weights:
def param_init_bdlayer(options, params, prefix='bd',
                       nin1=None, nin2=None, eye=False, bias=False):

    # params[_p(prefix, 'Md')] = 0.01 * numpy.random.randn((nin1,)).astype('float32')
    if eye:
        params[_p(prefix, 'Md')] = numpy.ones((nin1,), dtype='float32')
    else:
        print 'bd-eye init'
        params[_p(prefix, 'Md')] = norm_weight(nin1, 1, scale=0.01)[:, 0]


    if bias:
        params[_p(prefix, 'b')] = numpy.float32(0.0)  # 0.0

    return params


def bdlayer(tparams, input1, input2, cov=None, prefix='bd',
            activ='lambda x: tensor.nnet.sigmoid(x)', **kwargs):

    # if cov is not None:
    #     assert (_p(prefix, 'b') in tparams, 'coverage as bias')

    if input1.ndim == 2:
        input1 = input1 * tparams[_p(prefix, 'Md')][None, :]
        output = tensor.dot(input1, input2.dimshuffle(1, 0))
    else:
        input1 = input1 * tparams[_p(prefix, 'Md')][None, None, :]
        output = tensor.batched_dot(input1.dimshuffle(1, 0, 2),
                                    input2.dimshuffle(1, 2, 0))
        output = output.dimshuffle(1, 0, 2)  # dec_len x batch_size x enc_len

    if cov:
        output += tparams[_p(prefix, 'b')] * cov

    return eval(activ)(output)


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_, alpha_sum,
                    pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = alpha - alpha.max(axis=0, keepdims=True)  # safe attention >> fix a small bug
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha_sum = alpha.sum(axis=0)
        alpha = alpha / alpha_sum
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T, alpha_sum  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples)],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval



