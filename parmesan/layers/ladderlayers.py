import numpy as np
import lasagne
import theano.tensor as T
from lasagne.layers import MergeLayer, SliceLayer, GaussianNoiseLayer, NonlinearityLayer, DenseLayer
from lasagne import init
from lasagne import nonlinearities
from parmesan.layers.normalize import NormalizeLayer, ScaleAndShiftLayer
from parmesan.layers.special import ListIndexLayer


class DecoderNormalizeLayer(lasagne.layers.MergeLayer):
    """
        Special purpose layer used to construct the ladder network

        See the ladder_network example.
    """
    def __init__(self, incoming, mean, var, **kwargs):
        super(DecoderNormalizeLayer, self).__init__(
            [incoming, mean, var], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input, mean, var = inputs
        return (input - mean) / T.sqrt(var)


class DenoiseLayer(MergeLayer):
    """
        Special purpose layer used to construct the ladder network

        See the ladder_network example.
    """
    def __init__(self, u_net, z_net,
                 nonlinearity=nonlinearities.sigmoid, **kwargs):
        super(DenoiseLayer, self).__init__([u_net, z_net], **kwargs)

        u_shp, z_shp = self.input_shapes


        if not u_shp[-1] == z_shp[-1]:
            raise ValueError("last dimension of u and z  must be equal"
                             " u was %s, z was %s" % (str(u_shp), str(z_shp)))
        self.num_inputs = z_shp[-1]
        self.nonlinearity = nonlinearity
        constant = init.Constant
        self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
        self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
        self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
        self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

        self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
        self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
        self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

        self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

        self.b1 = self.add_param(constant(0.), (self.num_inputs,),
                                 name="b1", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])  # make a mutable copy
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        u, z_lat = inputs
        sigval = self.c1 + self.c2*z_lat
        sigval += self.c3*u + self.c4*z_lat*u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
        z_est += self.a3*u + self.a4*z_lat*u

        return z_est


def get_unlab(l, num_labels):
    return SliceLayer(l, indices=slice(num_labels, None), axis=0)


def decode_denoise(z_hat_in, z_noise, layer_index, num_labels):
    normalize = NormalizeLayer(z_hat_in, name='dec_normalize%i' % layer_index)
    u = ScaleAndShiftLayer(normalize, name='dec_scale%i' % layer_index)
    return DenoiseLayer(u_net=u, z_net=get_unlab(z_noise, num_labels), name='dec_denoise%i' % layer_index)


def decode_normalize(z_hat, norm_list, layer_index):
    mean = ListIndexLayer(norm_list, index=1, name='dec_index_mean%i' % layer_index)
    var = ListIndexLayer(norm_list, index=2, name='dec_index_var%i' % layer_index)
    return DecoderNormalizeLayer(z_hat, mean=mean, var=var,
                                 name='dec_decnormalize%i' % layer_index)


class RasmusInit(lasagne.init.Initializer):
    """Sample initial weights from the Gaussian distribution.
    Initial weight parameters are sampled from N(mean, std).
    Parameters

    https://github.com/arasmus/ladder
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=1.0, mean=0.0):
        self.std = std
        self.mean = mean

    # std one should reproduce rasmus init...
    def sample(self, shape):
        return lasagne.utils.floatX(lasagne.random.get_rng().normal(
            self.mean, self.std, size=shape) /
                      np.sqrt(shape[0]))



class AbstractLadderLayer (object):
    def create_encoder(self, incoming, unlabeled_slice, layer_num):
        raise NotImplementedError('abstract for class {0}'.format(type(self)))


    def create_decoder_2(self, z_hat_pre, z_noise, norm_list, num_labels, layer_num):
        raise NotImplementedError('abstract for class {0}'.format(type(self)))


class XformLadderLayer (AbstractLadderLayer):
    def __init__(self, nonlinearity, init=None, cost_weight=1.0, noise=0.3):
        super(XformLadderLayer, self).__init__()
        if init is None:
            init = RasmusInit()
        self.nonlinearity = nonlinearity
        self.init = init
        self.noise = noise
        self.cost_weight = cost_weight

    def _encode_xform(self, incoming, i):
        raise NotImplementedError('abstract for class {0}'.format(type(self)))

    def _decode_xform(self, z_hat_in, i):
        raise NotImplementedError('abstract for class {0}'.format(type(self)))

    def create_encoder(self, incoming, unlabeled_slice, layer_num):
        i = layer_num
        z_pre = self._encode_xform(incoming, i)
        norm_list = NormalizeLayer(
            z_pre, return_stats=True, name='enc_normalize%i' % i,
            stat_indices=unlabeled_slice)
        z = ListIndexLayer(norm_list, index=0, name='enc_index%i' % i)
        z_noise = GaussianNoiseLayer(z, sigma=self.noise, name='enc_noise%i' % i)
        h = NonlinearityLayer(
            ScaleAndShiftLayer(z_noise, name='enc_scale%i' % i),
            nonlinearity=self.nonlinearity, name='enc_nonlin%i' % i)
        return h, z, z_noise, norm_list


    def create_decoder_2(self, z_hat_pre, z_noise, norm_list, num_labels, layer_num):
        i = layer_num
        z_hat = decode_denoise(z_hat_pre, z_noise, i, num_labels)
        z_hat_bn = decode_normalize(z_hat, norm_list, i)
        z_hat_pre_below = self._decode_xform(z_hat, i - 1)
        return z_hat, z_hat_bn, z_hat_pre_below


class DenseLadderLayer (XformLadderLayer):
    def __init__(self, num_units_in, num_units_out, nonlinearity=lasagne.nonlinearities.rectify, init=None,
                 cost_weight=1.0, noise=0.3):
        super(DenseLadderLayer, self).__init__(nonlinearity=nonlinearity, init=init, cost_weight=cost_weight,
                                               noise=noise)
        self.num_units_in = num_units_in
        self.num_units_out = num_units_out

    def _encode_xform(self, incoming, i):
        z_pre = DenseLayer(
            incoming=incoming, num_units=self.num_units_out, nonlinearity=nonlinearities.identity, b=None,
            name='enc_dense%i' % i, W=self.init)
        return z_pre

    def _decode_xform(self, z_hat_in, i):
        z_hat_W = DenseLayer(z_hat_in, num_units=self.num_units_in, name='dec_dense%i' % i,
                             W=self.init, nonlinearity=nonlinearities.identity)
        return z_hat_W

class InputLadderLayer (AbstractLadderLayer):
    def __init__(self, shape, noise=0.3, cost_weight=1.0):
        super(InputLadderLayer, self).__init__()
        self.shape = shape
        self.noise = noise
        self.cost_weight = cost_weight

    def create_encoder(self, incoming, unlabeled_slice, layer_num):
        z = lasagne.layers.InputLayer(shape=self.shape)
        z_noise = GaussianNoiseLayer(z, sigma=self.noise, name='enc_noise%i' % layer_num)
        h = z_noise
        return h, z, z_noise, None


    def create_decoder_2(self, z_hat_pre, z_noise, norm_list, num_labels, layer_num):
        i = layer_num
        z_hat = decode_denoise(z_hat_pre, z_noise, i, num_labels=num_labels)
        z_hat_bn = z_hat   # for consistency
        return z_hat, z_hat_bn, None


