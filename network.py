
from typing import Any, List, Sequence, Tuple

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import utils
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape, BatchNormalization, LSTMCell
from tensorflow_probability.python.distributions import kullback_leibler

tfd = tfp.distributions
_NUM_FUNCTIONS = len(actions.FUNCTIONS)


class FullyConv(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(FullyConv, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)
    
    self.screen_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(24, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(36, 5, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu'),
    ])

    self.screen_input_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(24, 1, padding='same', activation='relu')
    ])

    self.single_select_encoder = tf.keras.layers.Dense(32, activation='relu')
    self.multi_select_encoder = tf.keras.layers.Dense(32, activation='relu')

    self.feature_fc_1 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_2 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_3 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_4 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_5 = tf.keras.layers.Dense(800, activation='relu')
    self.fn_out = tf.keras.layers.Dense(_NUM_FUNCTIONS)

    self.screen = tf.keras.Sequential()
    self.screen.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen.add(tf.keras.layers.Flatten())

    self.minimap = tf.keras.Sequential()
    self.minimap.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.minimap.add(tf.keras.layers.Flatten())

    self.screen2 = tf.keras.Sequential()
    self.screen2.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen2.add(tf.keras.layers.Flatten())

    self.queued = tf.keras.Sequential()
    self.queued.add(tf.keras.layers.Dense(2))

    self.control_group_act = tf.keras.Sequential()
    self.control_group_act.add(tf.keras.layers.Dense(5))

    self.control_group_id = tf.keras.Sequential()
    self.control_group_id.add(tf.keras.layers.Dense(10))

    self.select_point_act = tf.keras.Sequential()
    self.select_point_act.add(tf.keras.layers.Dense(4))

    self.select_add = tf.keras.Sequential()
    self.select_add.add(tf.keras.layers.Dense(2))

    self.select_unit_act = tf.keras.Sequential()
    self.select_unit_act.add(tf.keras.layers.Dense(4))

    self.select_unit_id = tf.keras.Sequential()
    self.select_unit_id.add(tf.keras.layers.Dense(500))

    self.select_worker = tf.keras.Sequential()
    self.select_worker.add(tf.keras.layers.Dense(4))

    self.build_queue_id = tf.keras.Sequential()
    self.build_queue_id.add(tf.keras.layers.Dense(10))

    self.unload_id = tf.keras.Sequential()
    self.unload_id.add(tf.keras.layers.Dense(500))

    self.dense2 = tf.keras.layers.Dense(1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out': self.args_out,
        'input_shape': self.input_shape
    })
    return config

  def call(self, feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue, 
           single_select, multi_select, score_cumulative, act_history, memory_state, carry_state):
    batch_size = tf.shape(feature_screen)[0]

    feature_screen_encoded = self.screen_encoder(feature_screen)

    single_select_encoded = self.single_select_encoder(single_select)
    single_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(single_select_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    single_select_encoded = tf.cast(single_select_encoded, 'float32')

    multi_select_encoded = self.multi_select_encoder(multi_select)
    multi_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(multi_select_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    multi_select_encoded = tf.cast(multi_select_encoded, 'float32')

    feature_encoded = tf.concat([feature_screen_encoded, single_select_encoded, multi_select_encoded], axis=3)

    feature_encoded_for_screen = self.screen_input_encoder(feature_encoded)

    screen_input = tf.keras.layers.ReLU()(feature_encoded_for_screen + feature_screen)
    minimap_input = feature_minimap

    feature_encoded_flatten = Flatten()(feature_encoded)
    feature_fc = self.feature_fc_1(feature_encoded_flatten)
    feature_fc = self.feature_fc_2(feature_fc)
    feature_fc = self.feature_fc_3(feature_fc)
    feature_fc = self.feature_fc_4(feature_fc)
    feature_fc = self.feature_fc_5(feature_fc)

    fn_out = self.fn_out(feature_fc)
    value = self.dense2(feature_fc)
    
    final_memory_state = memory_state
    final_carry_state = carry_state

    screen_args_out = self.screen(screen_input)
    minimap_args_out = self.minimap(minimap_input)
    screen2_args_out = self.screen2(screen_input)
    queued_args_out = self.queued(feature_fc)
    control_group_act_args_out = self.control_group_act(feature_fc)
    control_group_id_args_out = self.control_group_id(feature_fc)
    select_point_act_args_out = self.select_point_act(feature_fc)
    select_add_args_out = self.select_add(feature_fc)
    select_unit_act_args_out = self.select_unit_act(feature_fc)
    select_unit_id_args_out = self.select_unit_id(feature_fc)
    select_worker_args_out = self.select_worker(feature_fc)
    build_queue_id_args_out = self.build_queue_id(feature_fc)
    unload_id_args_out = self.unload_id(feature_fc)

    return fn_out, \
           screen_args_out, minimap_args_out, screen2_args_out, queued_args_out, control_group_act_args_out, control_group_id_args_out, \
           select_point_act_args_out, select_add_args_out, select_unit_act_args_out, select_unit_id_args_out, select_worker_args_out, \
           build_queue_id_args_out, unload_id_args_out, \
           value, final_memory_state, final_carry_state



def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True) 

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

  output = tf.matmul(attention_weights, v)  

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wk = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wv = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)
    self.dense = tf.keras.layers.Dense(d_model)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
    })
    return config
    
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training):
    batch_size = tf.shape(q)[0]
    
    v_original = v
    
    q = self.wq(q)  
    k = self.wk(k) 
    v = self.wv(v)  

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size) 

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])  
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) 

    output = self.dense(concat_attention)
    output = self.dense(output) 
    
    return output, attention_weights


class RelationalFullyConv(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(RelationalFullyConv, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)
    
    self.screen_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(47, 3, (2, 2), padding='same', activation='relu'),
       #tf.keras.layers.Conv2D(95, 3, (2, 2), padding='same', activation='relu')
    ])

    self.attention_screen_1 = MultiHeadAttention(48, 4)
    self.layernorm_screen_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout_screen_1 = tf.keras.layers.Dropout(0.1)

    self.attention_screen_2 = MultiHeadAttention(48, 4)
    self.layernorm_screen_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout_screen_2 = tf.keras.layers.Dropout(0.1)

    self._conv_out_size_screen = 8
    self._locs_screen = []
    for i in range(0, self._conv_out_size_screen*self._conv_out_size_screen):
        self._locs_screen.append(i / float(self._conv_out_size_screen*self._conv_out_size_screen))
        
    self._locs_screen = tf.expand_dims(self._locs_screen, 0)
    self._locs_screen = tf.expand_dims(self._locs_screen, 2)

    #self.minimap_encoder = tf.keras.Sequential([
    #   tf.keras.layers.Conv2D(8, 1, padding='same', activation='relu'),
    #   tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu'),
    #   tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    #])

    self.feature_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(24, 1, padding='same', activation='relu')
    ])

    #self.minimap_input_encoder = tf.keras.Sequential([
    #   tf.keras.layers.Conv2D(7, 1, padding='same', activation='relu')
    #])

    #self.minimap_encoder = tf.keras.Sequential([
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(256, activation='relu')
    #])

    #self.feature_decoder = tf.keras.Sequential([
    #   tf.keras.layers.Dense(32*32, activation='relu')
    #])

    #self.player_encoder = tf.keras.layers.Dense(11, activation='relu')
    #self.game_loop_encoder = tf.keras.layers.Dense(16, activation='relu')
    #self.available_actions_encoder = tf.keras.layers.Dense(32, activation='relu')
    #self.build_queue_encoder = tf.keras.layers.Dense(5, activation='relu')
    self.single_select_encoder = tf.keras.layers.Dense(16, activation='relu')
    #self.multi_select_encoder = tf.keras.layers.Dense(32, activation='relu')
    #self.score_cumulative_encoder = tf.keras.layers.Dense(10, activation='relu')
    #self.relational_nonspatial_encoder = tf.keras.layers.Dense(512, activation='relu')
    #self.encoding_lookup = utils.positional_encoding(max_position=2000, embedding_size=32)

    self.act_history_encoder = tf.keras.Sequential([
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(64, activation='relu'),
       #tf.keras.layers.Reshape((32, 32, 3), input_shape=(32*32*3,)),
       #tf.keras.layers.Conv2D(4, 4, 2, padding="same", activation="relu"),
       #tf.keras.layers.Conv2D(8, 2, 1, padding="same", activation="relu"),
       #tf.keras.layers.Conv2D(16, 3, 1, padding="same", activation="relu")
    ])

    self.screen_decoder = tf.keras.Sequential([
       #tf.keras.layers.Conv2DTranspose(filters=95, kernel_size=3, strides=2, padding='same', activation='relu'),
       tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, strides=2, padding='same', activation='relu'),
    ])

    self.feature_fc_1 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_2 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_3 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_4 = tf.keras.layers.Dense(800, activation='relu')
    self.feature_fc_5 = tf.keras.layers.Dense(800, activation='relu')
    self.fn_out = tf.keras.layers.Dense(_NUM_FUNCTIONS)

    self.screen = tf.keras.Sequential()
    self.screen.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen.add(tf.keras.layers.Flatten())

    self.minimap = tf.keras.Sequential()
    self.minimap.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.minimap.add(tf.keras.layers.Flatten())

    self.screen2 = tf.keras.Sequential()
    self.screen2.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen2.add(tf.keras.layers.Flatten())

    self.queued = tf.keras.Sequential()
    self.queued.add(tf.keras.layers.Dense(2))

    self.control_group_act = tf.keras.Sequential()
    self.control_group_act.add(tf.keras.layers.Dense(5))

    self.control_group_id = tf.keras.Sequential()
    self.control_group_id.add(tf.keras.layers.Dense(10))

    self.select_point_act = tf.keras.Sequential()
    self.select_point_act.add(tf.keras.layers.Dense(4))

    self.select_add = tf.keras.Sequential()
    self.select_add.add(tf.keras.layers.Dense(2))

    self.select_unit_act = tf.keras.Sequential()
    self.select_unit_act.add(tf.keras.layers.Dense(4))

    self.select_unit_id = tf.keras.Sequential()
    self.select_unit_id.add(tf.keras.layers.Dense(500))

    self.select_worker = tf.keras.Sequential()
    self.select_worker.add(tf.keras.layers.Dense(4))

    self.build_queue_id = tf.keras.Sequential()
    self.build_queue_id.add(tf.keras.layers.Dense(10))

    self.unload_id = tf.keras.Sequential()
    self.unload_id.add(tf.keras.layers.Dense(500))

    self.dense2 = tf.keras.layers.Dense(1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out': self.args_out,
        'input_shape': self.input_shape
    })
    return config

  def call(self, feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue, 
           single_select, multi_select, score_cumulative, act_history, memory_state, carry_state, training):
    batch_size = tf.shape(feature_screen)[0]

    #print("feature_screen.shape: " , feature_screen.shape)
    feature_screen_encoded = self.screen_encoder(feature_screen)
    #print("feature_screen_encoded.shape: " , feature_screen_encoded)
    # shape=(None, 8, 8, 95)

    feature_screen_encoded_attention = tf.reshape(feature_screen_encoded, 
                                                  [batch_size, self._conv_out_size_screen * self._conv_out_size_screen, 47])

    locs_screen = tf.tile(self._locs_screen, [batch_size, 1, 1])
    feature_screen_encoded_locs = tf.concat([feature_screen_encoded_attention, locs_screen], 2)

    #print("feature_screen_encoded_locs.shape: ", feature_screen_encoded_locs.shape)

    attention_feature_screen_1, _ = self.attention_screen_1(feature_screen_encoded_locs,
                                                            feature_screen_encoded_locs,
                                                            feature_screen_encoded_locs, None)
    attention_feature_screen_1 = self.dropout_screen_1(attention_feature_screen_1, training=training)
    attention_feature_screen_1 = self.layernorm_screen_1(feature_screen_encoded_locs + attention_feature_screen_1)

    attention_feature_screen_2, _ = self.attention_screen_2(attention_feature_screen_1,
                                                            attention_feature_screen_1,
                                                            attention_feature_screen_1, None)
    attention_feature_screen_2 = self.dropout_screen_2(attention_feature_screen_2, training=training)
    attention_feature_screen_2 = self.layernorm_screen_2(attention_feature_screen_1 + attention_feature_screen_2)

    #print("attention_feature_screen_2.shape: ", attention_feature_screen_2.shape)

    relational_spatial = tf.reshape(attention_feature_screen_2, [batch_size, 
                                                                 self._conv_out_size_screen, self._conv_out_size_screen, 48])
    relational_spatial = self.screen_decoder(relational_spatial)
    #print("relational_spatial.shape: ", relational_spatial.shape)

    #relational_nonspatial = tf.math.reduce_max(attention_feature_screen_2, 1)
    #relational_nonspatial = self.relational_nonspatial_encoder(relational_nonspatial)

    #print("relational_nonspatial.shape: ", relational_nonspatial.shape)

    #attention_feature_screen_encoded = tf.tile(tf.expand_dims(tf.expand_dims(attention_feature_screen, 1), 2),
    #                                           tf.stack([1, 32, 32, 1]))
    #attention_feature_screen_encoded = tf.cast(attention_feature_screen_encoded, 'float32')
    #print("attention_feature_screen_encoded.shape: ", attention_feature_screen_encoded.shape)

    #feature_minimap_encoded = self.minimap_encoder(feature_minimap)

    #print("single_select.shape: ", single_select.shape)
    single_select_encoded = self.single_select_encoder(single_select)
    single_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(single_select_encoded, 1), 2),
                                    tf.stack([1, self.screen_size, self.screen_size, 1]))
    single_select_encoded = tf.cast(single_select_encoded, 'float32')

    #multi_select_encoded = self.multi_select_encoder(multi_select)
    #multi_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(multi_select_encoded, 1), 2),
    #                                        tf.stack([1, self.screen_size, self.screen_size, 1]))
    #multi_select_encoded = tf.cast(multi_select_encoded, 'float32')

    #player_encoded = self.player_encoder(player)
    #print("player_encoded.shape: ", player_encoded.shape)
    #player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(player_encoded, 1), 2),
    #                                        tf.stack([1, 32, 32, 1]))
    #player_encoded = tf.cast(player_encoded, 'float32')

    #act_history = tf.expand_dims(act_history, 3)
    act_history_encoded = self.act_history_encoder(act_history)
    act_history_encoded = tf.tile(tf.expand_dims(tf.expand_dims(act_history_encoded, 1), 2),
                                  tf.stack([1, self.screen_size, self.screen_size, 1]))
    act_history_encoded = tf.cast(act_history_encoded, 'float32')
    #print("act_history_encoded.shape: ", act_history_encoded.shape)
    #print("")

    feature_spatial = tf.concat([relational_spatial, single_select_encoded, act_history_encoded], axis=3)
    feature_spatial_encoded = self.feature_encoder(feature_spatial)

    #feature_encoded_for_screen = self.screen_input_encoder(feature_encoded)
    #feature_encoded_for_minimap = self.minimap_input_encoder(feature_encoded)

    screen_input = tf.keras.layers.ReLU()(feature_spatial_encoded + feature_screen)
    #minimap_input = tf.keras.layers.ReLU()(feature_encoded_for_minimap + feature_minimap)
    minimap_input = feature_minimap

    feature_spatial_flatten = Flatten()(feature_spatial)
    feature_fc = self.feature_fc_1(feature_spatial_flatten)
    feature_fc = self.feature_fc_2(feature_fc)
    feature_fc = self.feature_fc_3(feature_fc)
    feature_fc = self.feature_fc_4(feature_fc)
    feature_fc = self.feature_fc_5(feature_fc)

    fn_out = self.fn_out(feature_fc)
    value = self.dense2(feature_fc)
    
    final_memory_state = memory_state
    final_carry_state = carry_state

    screen_args_out = self.screen(screen_input)
    minimap_args_out = self.minimap(minimap_input)
    screen2_args_out = self.screen2(screen_input)
    queued_args_out = self.queued(feature_fc)
    control_group_act_args_out = self.control_group_act(feature_fc)
    control_group_id_args_out = self.control_group_id(feature_fc)
    select_point_act_args_out = self.select_point_act(feature_fc)
    select_add_args_out = self.select_add(feature_fc)
    select_unit_act_args_out = self.select_unit_act(feature_fc)
    select_unit_id_args_out = self.select_unit_id(feature_fc)
    select_worker_args_out = self.select_worker(feature_fc)
    build_queue_id_args_out = self.build_queue_id(feature_fc)
    unload_id_args_out = self.unload_id(feature_fc)

    return fn_out, \
           screen_args_out, minimap_args_out, screen2_args_out, queued_args_out, control_group_act_args_out, control_group_id_args_out, \
           select_point_act_args_out, select_add_args_out, select_unit_act_args_out, select_unit_id_args_out, select_worker_args_out, \
           build_queue_id_args_out, unload_id_args_out, \
           value, final_memory_state, final_carry_state



def make_model(name):
    feature_screen = tf.keras.Input(shape=(16, 16, 24))
    feature_minimap = tf.keras.Input(shape=(16, 16, 7))
    player = tf.keras.Input(shape=(11))
    feature_units = tf.keras.Input(shape=(50, 8))
    available_actions = tf.keras.Input(shape=(573))
    memory_state = tf.keras.Input(shape=(256))
    carry_state = tf.keras.Input(shape=(256))
    game_loop = tf.keras.Input(shape=(1))
    build_queue = tf.keras.Input(shape=(5))
    single_select = tf.keras.Input(shape=(3))
    multi_select = tf.keras.Input(shape=(10))
    score_cumulative = tf.keras.Input(shape=(13))
    act_history = tf.keras.Input(shape=(16, utils._NUM_FUNCTIONS))

    if name == 'fullyconv':
        prediction = FullyConv(16, 16)(feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue, single_select, 
                                       multi_select, score_cumulative, act_history, memory_state, carry_state)
    elif name == 'relationalfullyconv':
        prediction = RelationalFullyConv(16, 16)(feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue, single_select, 
                                                 multi_select, score_cumulative, act_history, memory_state, carry_state)

    fn_out = prediction[0]

    screen_args_out = prediction[1]
    minimap_args_out = prediction[2]
    screen2_args_out = prediction[3]
    queued_args_out = prediction[4]
    control_group_act_args_out = prediction[5]
    control_group_id_args_out = prediction[6]
    select_point_act_args_out = prediction[7]
    select_add_args_out = prediction[8]
    select_unit_act_args_out = prediction[9]
    select_unit_id_args_out = prediction[10]
    select_worker_args_out = prediction[11]
    build_queue_id_args_out = prediction[12]
    unload_id_args_out = prediction[13]

    value = prediction[14]

    final_memory_state = prediction[15]
    final_carry_state = prediction[16]

    model = tf.keras.Model(inputs=[feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue, single_select, multi_select, 
                                   score_cumulative, act_history, memory_state, carry_state], 
                           outputs=[fn_out,
                                    screen_args_out, minimap_args_out, screen2_args_out, queued_args_out, control_group_act_args_out, control_group_id_args_out,
                                    select_point_act_args_out, select_add_args_out, select_unit_act_args_out, select_unit_id_args_out, select_worker_args_out,
                                    build_queue_id_args_out, unload_id_args_out,
                                    value, final_memory_state, final_carry_state],
                           name=name)
    return model