import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Conv2D, Flatten, Lambda, Reshape, Add, Layer
from reaver.models.base.layers import Squeeze, Split, Transpose, Log, Broadcast2D
from tensorflow.keras import backend as K


@gin.configurable
def build_graph_network(obs_spec, act_spec, data_format='channels_first', broadcast_non_spatial=False, fc_dim=256):
    screen, screen_input = transformer_block('screen', obs_spec.spaces[0], conv_cfg(data_format, 'relu'))
    non_spatial_inputs = [Input(s.shape) for s in obs_spec.spaces[1:]]

    if broadcast_non_spatial:
        non_spatial, spatial_dim = non_spatial_inputs[1], obs_spec.spaces[0].shape[1]
        non_spatial = Log()(non_spatial)
        broadcasted_non_spatial = Broadcast2D(spatial_dim)(non_spatial)
        state = Concatenate(axis=1, name="state_block")([screen, broadcasted_non_spatial])
    else:
        state = screen

    fc = Flatten(name="state_flat")(state)
    fc = Dense(fc_dim, **dense_cfg('relu'))(fc)

    value = Dense(1, name="value_out", **dense_cfg(scale=0.1))(fc)
    value = Squeeze(axis=-1)(value)

    logits = []
    for space in act_spec:
        if space.is_spatial():
            fc_temp = Flatten()(state)
            fc_temp = Dense(32 * 32, **dense_cfg('relu'))(fc_temp)
            logits.append(fc_temp)
        else:
            logits.append(Dense(space.size(), **dense_cfg(scale=0.1))(fc))
    
    mask_actions = Lambda(
        lambda x: tf.where(non_spatial_inputs[0] > 0, x, -1000 * tf.ones_like(x)),
        name="mask_unavailable_action_ids"
    )

    logits[0] = mask_actions(logits[0])
    return Model(
        inputs = [screen_input] + non_spatial_inputs,
        outputs = logits + [value]
    )


class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape


def transformer_block(name, space, cfg):
    inpt = Input(space.shape, name=name + '_input')
    block = Split(space.shape[0], axis=1)(inpt)

    for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):
        if dim > 1:
            block[i] = Squeeze(axis=1)(block[i])
            # Embedding dim 10 as per https://arxiv.org/pdf/1806.01830.pdf
            block[i] = Embedding(input_dim=dim, output_dim=10)(block[i])
            # [N, H, W, C] -> [N, C, H, W]
            block[i] = Transpose([0, 3, 1, 2])(block[i])
        else:
            block[i] = Log()(block[i])

    block = Concatenate(axis=1)(block)

    cfg_temp = dict(
        padding='same',
        activation='relu',
        data_format='channels_first',
        kernel_initializer=VarianceScaling(scale=2.0*1.0)
    )
    block_temp = Conv2D(16, 3, **cfg_temp)(block)
    block_temp = Conv2D(32, 3, **cfg_temp)(block_temp)
    block_temp = Transpose([0,2,3,1])(block_temp)
    block_temp = Reshape([32*32,32])(block_temp)
   
    l = 32*32
    d = 8*4
    dv = 8
    dout = 8*4
    nv = 4

    v2 = Dense(dv*nv, activation = "relu")(block_temp)
    q2 = Dense(dv*nv, activation = "relu")(block_temp)
    k2 = Dense(dv*nv, activation = "relu")(block_temp)

    v = Reshape([l,nv,dv])(v2)
    q = Reshape([l,nv,dv])(q2)
    k = Reshape([l,nv,dv])(k2)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1], axes=[-1,-1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q,k])
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4,3]), output_shape=(l, nv, dv))([att, v])
    out = Reshape([l,d])(out)
    out = Add()([out,block_temp])
    out = Dense(dout, activation = "relu")(out)
    out = Reshape([32,32,32])(out)
    x = NormL()(out)

    return x, inpt


def conv_cfg(data_format='channels_first', activation=None, scale=1.0):
    return dict(
        padding='same',
        activation=activation,
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0*scale)
    )


def dense_cfg(activation=None, scale=1.0):
    return dict(
        activation=activation,
        kernel_initializer=VarianceScaling(scale=2.0*scale)
    )
