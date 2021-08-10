from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow_probability.python.distributions import kullback_leibler

tfd = tfp.distributions

_NUM_FUNCTIONS = len(actions.FUNCTIONS)


class ScalarEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarEncoder, self).__init__()
    self.output_dim = output_dim

    self.network = tf.keras.Sequential([
       tf.keras.layers.Dense(self.output_dim, activation='relu', name="ScalarEncoder_dense", kernel_regularizer='l2')
    ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, scalar_feature):
    batch_size = tf.shape(scalar_feature)[0]
    scalar_feature_encoded = self.network(scalar_feature)
    return scalar_feature_encoded


class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, height, width, channel):
    super(SpatialEncoder, self).__init__()

    self.height = height
    self.width = width

    self.network = tf.keras.Sequential([
       tf.keras.layers.Conv2D(13, 1, padding='same', activation='relu', name="SpatialEncoder_cond2d_1", kernel_regularizer='l2'),
       tf.keras.layers.Conv2D(int(height/2), 5, padding='same', activation='relu', name="SpatialEncoder_cond2d_2", 
                              kernel_regularizer='l2'),
       tf.keras.layers.Conv2D(height, 3, padding='same', activation='relu', name="SpatialEncoder_cond2d_3", kernel_regularizer='l2')
    ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, spatial_feature):
    spatial_feature_encoded = self.network(spatial_feature)

    return spatial_feature_encoded


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


class EntityEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim, entity_num):
    super(EntityEncoder, self).__init__()
    self.output_dim = output_dim

    self.attention = MultiHeadAttention(8, 1)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)

    self.entity_num = entity_num
    self.locs = []
    for i in range(0, self.entity_num):
        self.locs.append(i / float(self.entity_num))
            
    self.locs = tf.expand_dims(self.locs, 0)
    self.locs = tf.expand_dims(self.locs, 2)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim,
        'entity_num': self.entity_num
    })
    return config

  def call(self, entity_features, training):
    batch_size = tf.shape(entity_features)[0]

    locs = tf.tile(self.locs, [batch_size, 1, 1])
    entity_features_locs = tf.concat([entity_features, locs], 2)
    attention_output, _ = self.attention(entity_features_locs, entity_features_locs, entity_features_locs, None)
        
    attention_output = self.dropout(attention_output, training=training)
    attention_output = self.layernorm(entity_features_locs + attention_output)
    max_pool_1d = tf.math.reduce_max(attention_output, 1)
    output = max_pool_1d

    return output


class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number, network_scale):
    super(Core, self).__init__()

    self.unit_number = unit_number
    self.network_scale = network_scale

    self.lstm = LSTM(256*self.network_scale*self.network_scale, name="core_lstm", return_sequences=True, 
                     return_state=True, kernel_regularizer='l2')

    self.network = tf.keras.Sequential([Reshape((144, 256*self.network_scale*self.network_scale)),
                                        Flatten(),
                                        tf.keras.layers.Dense(256*self.network_scale*self.network_scale, activation='relu', 
                                                              name="core_dense", 
                                                              kernel_regularizer='l2')
                                       ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_number': self.unit_number,
        'network_scale': self.network_scale
    })
    return config

  def call(self, feature_encoded, memory_state, carry_state):
    batch_size = tf.shape(feature_encoded)[0]

    feature_encoded_flattened = Flatten()(feature_encoded)
    feature_encoded_flattened = Reshape((144, 256*self.network_scale*self.network_scale))(feature_encoded_flattened)

    core_output, final_memory_state, final_carry_state = self.lstm(feature_encoded_flattened, initial_state=(memory_state, carry_state))

    core_output = self.network(core_output)

    return core_output, final_memory_state, final_carry_state


class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, output_dim, network_scale):
    super(ActionTypeHead, self).__init__()

    self.output_dim = output_dim
    self.network_scale = network_scale
    self.network = tf.keras.Sequential([tf.keras.layers.Dense(self.output_dim, name="ActionTypeHead_dense_1", kernel_regularizer='l2'),
                                            tf.keras.layers.Softmax()])
    self.autoregressive_embedding_network = tf.keras.Sequential([tf.keras.layers.Dense(256*self.network_scale*self.network_scale, 
                                                                                       activation='relu', name="ActionTypeHead_dense_2", 
                                                                                       kernel_regularizer='l2')])
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim,
        'network_scale': self.network_scale
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]
    action_type_logits = self.network(core_output)

    action_type_dist = tfd.Categorical(probs=action_type_logits)
    action_type = action_type_dist.sample()
    action_type_onehot = tf.one_hot(action_type, self.output_dim)

    autoregressive_embedding = self.autoregressive_embedding_network(action_type_onehot)
    autoregressive_embedding += core_output

    return action_type_logits, autoregressive_embedding


class SpatialArgumentHead(tf.keras.layers.Layer):
  def __init__(self, height, width, channel):
    super(SpatialArgumentHead, self).__init__()

    self.height = height
    self.width = width
    self.network = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, padding='same', name="SpatialArgumentHead_conv2d_2", 
                                        kernel_regularizer='l2'),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Softmax()])

    self.autoregressive_embedding_encoder_1 = tf.keras.Sequential([tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                          		    name="SpatialArgumentHead_dense_2", 
                                                                          		    kernel_regularizer='l2')])
    self.autoregressive_embedding_encoder_2 = tf.keras.Sequential([tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                          		    name="SpatialArgumentHead_dense_4", 
                                                                          		    kernel_regularizer='l2')])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, feature_encoded, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    encoded_core_output = self.autoregressive_embedding_encoder_1(core_output)
    encoded_core_output = tf.reshape(encoded_core_output, (batch_size, self.height, self.width, 1))

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder_2(autoregressive_embedding)
    encoded_autoregressive_embedding = tf.reshape(encoded_autoregressive_embedding, (batch_size, self.height, self.width, 1))

    network_input = tf.concat([feature_encoded, encoded_core_output, encoded_autoregressive_embedding], axis=3)
    action_logits = self.network(network_input)

    return action_logits


class ScalarArgumentHead(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarArgumentHead, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.Sequential()
    self.network.add(tf.keras.layers.Dense(output_dim, name="ScalarArgumentHead_dense_2", kernel_regularizer='l2'))
    self.network.add(tf.keras.layers.Softmax())

    self.autoregressive_embedding_encoder = tf.keras.Sequential([tf.keras.layers.Dense(self.output_dim, activation='relu', 
                                                                          		  name="ScalarArgumentHead_dense_6", 
                                                                          		  kernel_regularizer='l2')
                                                                ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder(autoregressive_embedding)

    network_input = tf.concat([core_output, encoded_autoregressive_embedding], axis=1)
    action_logits = self.network(network_input)
    
    return action_logits


class Baseline(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(Baseline, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.Sequential([tf.keras.layers.Dense(1, name="Baseline_dense", kernel_regularizer='l2')])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]
    network_input = core_output
    value = self.network(network_input)

    return value


class AlphaStar(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(AlphaStar, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)

    # State Encoder
    self.player_encoder = ScalarEncoder(output_dim=11)
    self.screen_encoder = SpatialEncoder(height=screen_size, width=screen_size, channel=3)
    #self.entity_encoder = EntityEncoder(output_dim=11, entity_num=50)
    #self.available_actions_encoder = ScalarEncoder(output_dim=11)

    # Core
    self.core = Core(256, self.network_scale)

    # Action Head
    self.action_type_head = ActionTypeHead(_NUM_FUNCTIONS, self.network_scale)
    self.screen_argument_head = SpatialArgumentHead(height=screen_size, width=screen_size, channel=3)
    self.minimap_argument_head = SpatialArgumentHead(height=screen_size, width=screen_size, channel=3)
    self.screen2_argument_head = SpatialArgumentHead(height=screen_size, width=screen_size, channel=3)
    self.queued_argument_head = ScalarArgumentHead(2)
    self.control_group_act_argument_head = ScalarArgumentHead(5)
    self.control_group_id_argument_head = ScalarArgumentHead(10)
    self.select_point_act_argument_head = ScalarArgumentHead(4)
    self.select_add_argument_head = ScalarArgumentHead(2)
    self.select_unit_act_argument_head = ScalarArgumentHead(4)
    self.select_unit_id_argument_head = ScalarArgumentHead(500)
    self.select_worker_argument_head = ScalarArgumentHead(4)
    self.build_queue_id_argument_head = ScalarArgumentHead(10)
    self.unload_id_argument_head = ScalarArgumentHead(50)


    self.baseline = Baseline(256)
    self.args_out_logits = dict()

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out_logits': self.args_out_logits
    })
    return config

  def call(self, feature_screen, feature_player, feature_units, memory_state, carry_state, game_loop, available_actions, last_action_type):
    batch_size = tf.shape(feature_screen)[0]

    #feature_screen_array.shape:  (1, 32, 32, 28)
    #feature_player_array.shape:  (1, 32, 32, 11)
    feature_screen_encoded = self.screen_encoder(feature_screen)

    feature_player_encoded = self.player_encoder(feature_player)
    feature_player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_player_encoded, 1), 2),
                                         tf.stack([1, self.screen_size, self.screen_size, 1]))
    feature_player_encoded = tf.cast(feature_player_encoded, 'float32')

    #feature_units_encoded = self.entity_encoder(feature_units)
    #feature_units_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_units_encoded, 1), 2),
    #                                    tf.stack([1, self.screen_size, self.screen_size, 1]))
    #feature_units_encoded = tf.cast(feature_units_encoded, 'float32')

    game_loop_encoded = tf.tile(tf.expand_dims(tf.expand_dims(game_loop, 1), 2),
                                   tf.stack([1, self.screen_size, self.screen_size, 1]))
    game_loop_encoded = tf.cast(game_loop_encoded, 'float32')

    last_action_type_encoded = last_action_type / _NUM_FUNCTIONS
    last_action_type_encoded = tf.tile(tf.expand_dims(tf.expand_dims(last_action_type_encoded, 1), 2),
                                           tf.stack([1, self.screen_size, self.screen_size, 1]))
    last_action_type_encoded = tf.cast(last_action_type_encoded, 'float32')

    feature_encoded = tf.concat([feature_screen_encoded, feature_player_encoded, last_action_type_encoded], axis=3)
    
    core_output, final_memory_state, final_carry_state = self.core(feature_encoded, memory_state, carry_state)

    action_type_logits, autoregressive_embedding = self.action_type_head(core_output)
    
    for arg_type in actions.TYPES:
      if arg_type.name == 'screen':
        self.args_out_logits[arg_type] = self.screen_argument_head(feature_screen_encoded, core_output, autoregressive_embedding)
      elif arg_type.name == 'minimap':
        self.args_out_logits[arg_type] = self.minimap_argument_head(feature_screen_encoded, core_output, autoregressive_embedding)
      elif arg_type.name == 'screen2':
        self.args_out_logits[arg_type] = self.screen2_argument_head(feature_screen_encoded, core_output, autoregressive_embedding)
      elif arg_type.name == 'queued':
        self.args_out_logits[arg_type] = self.queued_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'control_group_act':
        self.args_out_logits[arg_type] = self.control_group_act_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'control_group_id':
        self.args_out_logits[arg_type] = self.control_group_id_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'select_point_act':
        self.args_out_logits[arg_type] = self.select_point_act_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'select_add':
        self.args_out_logits[arg_type] = self.select_add_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'select_unit_act':
        self.args_out_logits[arg_type] = self.select_unit_act_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'select_unit_id':
        self.args_out_logits[arg_type] = self.select_unit_id_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'select_worker':
        self.args_out_logits[arg_type] = self.select_worker_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'build_queue_id':
        self.args_out_logits[arg_type] = self.build_queue_id_argument_head(core_output, autoregressive_embedding)
      elif arg_type.name == 'unload_id':
        self.args_out_logits[arg_type] = self.unload_id_argument_head(core_output, autoregressive_embedding)


    value = self.baseline(core_output)

    return action_type_logits, self.args_out_logits, value, final_memory_state, final_carry_state
    
    
class FullyConvLSTM(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(FullyConvLSTM, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)

    self.screen_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(13, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    ])
    
    '''
    self.screen_encoder = tf.keras.Sequential()
    self.screen_encoder.add(ConvLSTM2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', return_sequences=True))
    self.screen_encoder.add(BatchNormalization())
    self.screen_encoder.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', return_sequences=True))
    self.screen_encoder.add(BatchNormalization())
    #self.screen_encoder.add(ConvLSTM2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', return_sequences=True))
    self.screen_encoder.add(Conv3D(filters=1, kernel_size=(2, 2, 2), activation="relu", padding="same"))
    '''
    self.feature_fc = tf.keras.layers.Dense(256, activation='relu')
    self.fn_out = tf.keras.layers.Dense(_NUM_FUNCTIONS, activation='softmax')

    self.screen = tf.keras.Sequential()
    self.screen.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen.add(tf.keras.layers.Flatten())
    self.screen.add(tf.keras.layers.Softmax())

    self.minimap = tf.keras.Sequential()
    self.minimap.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.minimap.add(tf.keras.layers.Flatten())
    self.minimap.add(tf.keras.layers.Softmax())

    self.screen2 = tf.keras.Sequential()
    self.screen2.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen2.add(tf.keras.layers.Flatten())
    self.screen2.add(tf.keras.layers.Softmax())

    self.queued = tf.keras.Sequential()
    self.queued.add(tf.keras.layers.Dense(2))
    self.queued.add(tf.keras.layers.Softmax())

    self.control_group_act = tf.keras.Sequential()
    self.control_group_act.add(tf.keras.layers.Dense(5))
    self.control_group_act.add(tf.keras.layers.Softmax())

    self.control_group_id = tf.keras.Sequential()
    self.control_group_id.add(tf.keras.layers.Dense(10))
    self.control_group_id.add(tf.keras.layers.Softmax())

    self.select_point_act = tf.keras.Sequential()
    self.select_point_act.add(tf.keras.layers.Dense(4))
    self.select_point_act.add(tf.keras.layers.Softmax())

    self.select_add = tf.keras.Sequential()
    self.select_add.add(tf.keras.layers.Dense(2))
    self.select_add.add(tf.keras.layers.Softmax())

    self.select_unit_act = tf.keras.Sequential()
    self.select_unit_act.add(tf.keras.layers.Dense(4))
    self.select_unit_act.add(tf.keras.layers.Softmax())

    self.select_unit_id = tf.keras.Sequential()
    self.select_unit_id.add(tf.keras.layers.Dense(500))
    self.select_unit_id.add(tf.keras.layers.Softmax())

    self.select_worker = tf.keras.Sequential()
    self.select_worker.add(tf.keras.layers.Dense(4))
    self.select_worker.add(tf.keras.layers.Softmax())

    self.build_queue_id = tf.keras.Sequential()
    self.build_queue_id.add(tf.keras.layers.Dense(10))
    self.build_queue_id.add(tf.keras.layers.Softmax())

    self.unload_id = tf.keras.Sequential()
    self.unload_id.add(tf.keras.layers.Dense(500))
    self.unload_id.add(tf.keras.layers.Softmax())

    self.dense2 = tf.keras.layers.Dense(1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out': self.args_out,
        'input_shape': self.input_shape
    })
    return config

  def call(self, feature_screen, feature_player, feature_units, game_loop, available_actions, last_action_type):
    batch_size = tf.shape(feature_screen)[0]

    #feature_screen_array.shape:  (1, 32, 32, 13)
    #feature_player_array.shape:  (1, 32, 32, 11)
    feature_screen_encoded = self.screen_encoder(feature_screen)
    #feature_screen_flatten = Flatten()(feature_screen_encoded)
    #feature_screen_reshaped = Reshape((32, 32, 4))(feature_screen_flatten)

    feature_player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_player, 1), 2),
                                     tf.stack([1, 32, 32, 1]))
    feature_player_encoded = tf.cast(feature_player_encoded, 'float32')

    last_action_type_encoded = last_action_type / _NUM_FUNCTIONS
    last_action_type_encoded = tf.tile(tf.expand_dims(tf.expand_dims(last_action_type_encoded, 1), 2),
                                       tf.stack([1, self.screen_size, self.screen_size, 1]))
    last_action_type_encoded = tf.cast(last_action_type_encoded, 'float32')

    feature_encoded = tf.concat([feature_screen_encoded, feature_player_encoded, last_action_type_encoded], axis=3)
    #feature_encoded_reshaped = Reshape((64, 128))(feature_encoded)

    feature_encoded_flatten = Flatten()(feature_encoded)
    feature_fc = self.feature_fc(feature_encoded_flatten)

    fn_out = self.fn_out(feature_fc)
    
    args_out = dict()
    for arg_type in actions.TYPES:
      if arg_type.name == 'screen':
        args_out[arg_type] = self.screen(feature_encoded)
      elif arg_type.name == 'minimap':
        args_out[arg_type] = self.minimap(feature_encoded)
      elif arg_type.name == 'screen2':
        args_out[arg_type] = self.screen2(feature_encoded)
      elif arg_type.name == 'queued':
        args_out[arg_type] = self.queued(feature_fc)
      elif arg_type.name == 'control_group_act':
        args_out[arg_type] = self.control_group_act(feature_fc)
      elif arg_type.name == 'control_group_id':
        args_out[arg_type] = self.control_group_id(feature_fc)
      elif arg_type.name == 'select_point_act':
        args_out[arg_type] = self.select_point_act(feature_fc)
      elif arg_type.name == 'select_add':
        args_out[arg_type] = self.select_add(feature_fc)
      elif arg_type.name == 'select_unit_act':
        args_out[arg_type] = self.select_unit_act(feature_fc)
      elif arg_type.name == 'select_unit_id':
        args_out[arg_type] = self.select_unit_id(feature_fc)
      elif arg_type.name == 'select_worker':
        args_out[arg_type] = self.select_worker(feature_fc)
      elif arg_type.name == 'build_queue_id':
        args_out[arg_type] = self.build_queue_id(feature_fc)
      elif arg_type.name == 'unload_id':
        args_out[arg_type] = self.unload_id(feature_fc)

    value = self.dense2(feature_fc)

    return fn_out, args_out, value


def make_model(name):
    '''
    feature_screen.shape:  (1, 32, 32, 28)
    feature_player.shape:  (1, 3)
    feature_units.shape:  (1, 50, 7)
    game_loop.shape:  (1, 1)
    available_actions.shape:  (1, 573)
    last_action_type.shape:  (1, 1)
    feature_screen_history.shape:  (1, 4, 32, 32, 2)
    '''
    feature_screen = tf.keras.Input(shape=(32, 32, 28))
    feature_player = tf.keras.Input(shape=(11))
    feature_units = tf.keras.Input(shape=(50, 7))
    game_loop = tf.keras.Input(shape=(1))
    available_actions = tf.keras.Input(shape=(573))
    last_action_type = tf.keras.Input(shape=(1))
    #feature_screen_history = tf.keras.Input(shape=(4, 32, 32, 13))

    fn_out, args_out, value = FullyConvLSTM(32, 32)(feature_screen, feature_player, feature_units, game_loop, available_actions, 
                                                    last_action_type)
    model = tf.keras.Model(inputs={'feature_screen': feature_screen, 'feature_player': feature_player, 
                                   'feature_units': feature_units, 'game_loop': game_loop,
                                   'available_actions': available_actions, 'last_action_type': last_action_type}, 
                           outputs={'fn_out': fn_out, 'args_out': args_out, 'value':value}, 
                           name=name)
    return model
