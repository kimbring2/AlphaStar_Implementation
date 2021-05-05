from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.env.environment import TimeStep, StepType

import sys
import numpy as np
import math
import random
import pylab
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def get_entity_obs(feature_units):
    unit_type = []
    alliance = []
    current_health = []
    x_position = []
    y_position = []
    is_selected = []

    for unit in feature_units:
        unit_type_ = unit.unit_type
        unit_type_onehot = np.identity(256)[unit_type_:unit_type_+1]

        unit_alliance = unit.alliance
        unit_alliance_onehot = np.identity(5)[unit_alliance:unit_alliance+1]
        unit_health = int(math.sqrt(unit.health))
        unit_health_onehot = np.identity(39)[unit_health:unit_health+1]

        x_position.append(bin_array(abs(unit.x), 10))
        y_position.append(bin_array(abs(unit.y), 10))

        is_selected_onehot = np.identity(2)[unit.is_selected:unit.is_selected+1]
  
        unit_type.append(unit_type_onehot[0])
        alliance.append(unit_alliance_onehot[0])
        current_health.append(unit_health_onehot[0])
        is_selected.append(is_selected_onehot[0])
    
    input_list = []
    length = len(feature_units)
    if length > 19:
        length = 19

    for i in range(0, length):
        entity_array = np.concatenate((unit_type[i], current_health[i], x_position[i], y_position[i], is_selected[i]), axis=0, out=None)
        input_list.append(entity_array)

    if length < 20:
        for i in range(length, 20):
            input_list.append(np.zeros(317))
 
    input_array = np.array(input_list)

    return input_array


def get_action(non_spatial_action_logits, selected_units_logits, spatial_action1_logits, spatial_action2_logits, feature_units, epsilon):
    #epsilon = 0.1
    if random.random() <= epsilon:
        non_spatial_action = tf.argmax(non_spatial_action_logits, axis=1)    
    else:
        non_spatial_action = random.randint(0, 6 - 1)
    #print("non_spatial_action_logits: " + str(non_spatial_action_logits))
    #non_spatial_action = np.random.choice(6, p=non_spatial_action_logits[0])

    if random.random() <= epsilon:
        spatial_action1 = tf.argmax(spatial_action1_logits, axis=1)
    else:
        spatial_action1 = random.randint(0, 16384 - 1)
    #spatial_action1 = np.random.choice(16384, p=spatial_action1_logits[0])

    if random.random() <= epsilon:
        spatial_action2 = tf.argmax(spatial_action2_logits, axis=1)  
    else:
        spatial_action2 = random.randint(0, 16384 - 1)
    #spatial_action2 = np.random.choice(16384, p=spatial_action2_logits[0])    

    if random.random() <= epsilon:
        selected_unit_index = tf.argmax(selected_units_logits, axis=1).numpy()[0]
    else:
        selected_unit_index = random.randint(0, len(feature_units) - 1)

    #print("non_spatial_action: " + str(non_spatial_action))
    self_units = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
    enermy_units = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

    if selected_unit_index > len(feature_units) - 1:
        selected_unit = random.choice(self_units)
    else:
        selected_unit = feature_units[selected_unit_index]

    spatial_action1_x = int(spatial_action1 / 128)
    spatial_action1_y = int(spatial_action1 % 128)

    spatial_action2_x = int(spatial_action2 / 128)
    spatial_action2_y = int(spatial_action2 % 128)

    #non_spatial_action = 0
    if non_spatial_action == 0:
        action = actions.FUNCTIONS.no_op()    
    elif non_spatial_action == 1:
        action = actions.FUNCTIONS.select_point("select", [int(selected_unit.x), int(selected_unit.y)])
    elif non_spatial_action == 2:
        action = actions.FUNCTIONS.select_unit("deselect_all_type", [selected_unit_index])
    elif non_spatial_action == 3:
        action = actions.FUNCTIONS.Move_screen("now", [spatial_action1_x, spatial_action1_y])
    elif non_spatial_action == 4:
        action = actions.FUNCTIONS.Attack_screen("now", [spatial_action1_x, spatial_action1_y])
    elif non_spatial_action == 5:
        action = actions.FUNCTIONS.select_rect("select", [spatial_action1_x, spatial_action1_y], [spatial_action2_x, spatial_action2_y])

    #print("action: " + str(action))
    return action


class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, img_height, img_width, channel):
    super(SpatialEncoder, self).__init__()

    self.img_height = img_height
    self.img_width = img_width

    self.map_model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(4, 2, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(16, 2, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(),
    ])

    self.dense = tf.keras.layers.Dense(256, activation='relu')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'img_height': self.img_height,
        'img_width': self.img_width,
    })
    return config

  def call(self, feature_screen):
    map = self.map_model(feature_screen)

    map_flatten = tf.keras.layers.Flatten()(map)
    map_flatten = tf.cast(map_flatten, tf.float32) 

    embedded_spatial = self.dense(map_flatten)

    return map, embedded_spatial


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  #dk = tf.shape(tf.shape(k)[-1])
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)

    self.conv1d = tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=(20,317))
    
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = tf.cast(q, tf.float32)
    k = tf.cast(k, tf.float32)
    v = tf.cast(v, tf.float32)

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    input_shape = concat_attention.shape
    output = self.conv1d(concat_attention)
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


class EntityEncoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, rate=0.1):
    super(EntityEncoder, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    self.mha = MultiHeadAttention(d_model=464, num_heads=8)
    self.conv1d = tf.keras.layers.Conv1D(256, 1, activation='relu')
    self.dense = tf.keras.layers.Dense(256, activation='relu')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
    })
    return config

  def call(self, embedded_feature_units):
    out, attn = self.mha(v=embedded_feature_units, k=embedded_feature_units, q=embedded_feature_units, mask=None)
    entity_embeddings = self.conv1d(out)

    embedded_entity = tf.reduce_mean(out, 1)
    embedded_entity = tf.cast(embedded_entity, tf.float32) 
    embedded_entity = self.dense(embedded_entity)

    return embedded_entity, entity_embeddings


class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number):
    super(Core, self).__init__()
    self.unit_number = unit_number

    self.lstm = tf.keras.layers.LSTM(unit_number, return_sequences=True)
    self.dense = tf.keras.layers.Dense(128*2, activation='relu')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_number': self.unit_number
    })
    return config

  def call(self, embedded_entity, embedded_spatial):
    batch_size = tf.shape(embedded_entity)[0]

    core_input = tf.concat((embedded_spatial, embedded_entity), axis=1)
    core_input = self.dense(core_input)
    core_input = tf.reshape(core_input, (batch_size, -1, 128*2))

    lstm_output = self.lstm(core_input, training=True)

    return lstm_output


def sample(a):
    return tf.argmax(a, axis=1)


class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, action_num):
    super(ActionTypeHead, self).__init__()

    self.action_num = action_num
    self.dense1 = tf.keras.layers.Dense(self.action_num)
    self.dense2 = tf.keras.layers.Dense(self.action_num, activation='softmax')
    self.dense3 = tf.keras.layers.Dense(512)
    self.dense4 = tf.keras.layers.Dense(512)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'action_num': self.action_num
    })
    return config

  def call(self, lstm_output):
    batch_size = tf.shape(lstm_output)[0]
    out = self.dense1(lstm_output)

    action_type_logits = self.dense2(out)
    action_type_logits = tf.reduce_mean(action_type_logits, axis=1)

    action_type = sample(action_type_logits)
    action_type_onehot = tf.one_hot(action_type, self.action_num)
    action_type_onehot = tf.cast(action_type_onehot, tf.float32) 

    autoregressive_embedding = self.dense3(action_type_onehot)
    lstm_output_embedding = tf.cast(lstm_output, tf.float32) 
    lstm_output_embedding = self.dense4(lstm_output_embedding)

    autoregressive_embedding += lstm_output_embedding

    return action_type_logits, autoregressive_embedding


class SelectedUnitsHead(tf.keras.layers.Layer):
  def __init__(self, unit_num):
    super(SelectedUnitsHead, self).__init__()
    self.unit_num = unit_num

    self.conv1d = tf.keras.layers.Conv1D(unit_num, 1, activation='relu')
    self.dense1 = tf.keras.layers.Dense(unit_num, activation='relu')
    self.dense2 = tf.keras.layers.Dense(unit_num, activation='relu')
    self.dense3 = tf.keras.layers.Dense(unit_num, activation='relu')
    self.lstm = tf.keras.layers.LSTM(units=unit_num, activation='softmax', return_state=True, return_sequences=True)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_num': self.unit_num
    })
    return config

  def call(self, autoregressive_embedding, action_type_logits, entity_embeddings):
    action_type = tf.argmax(action_type_logits, axis=1)
    func_embed = tf.one_hot(action_type, self.unit_num)
    func_embed = self.dense1(func_embed)

    key = self.conv1d(entity_embeddings)

    autoregressive_embedding = tf.cast(autoregressive_embedding, tf.float32) 
    query = self.dense2(autoregressive_embedding)
    query = query + func_embed

    batch_size = tf.shape(entity_embeddings)[0]
    dim = tf.zeros([batch_size, self.unit_num])
    query, state_h, state_c = self.lstm(query, initial_state=[dim, dim], training=True)
    selected_units_logits = tf.matmul(query, key, transpose_b=True)
    selected_units_logits = tf.reduce_mean(selected_units_logits, axis=1)
    selected_units = sample(selected_units_logits)

    selected_units_embedding = tf.one_hot(selected_units, self.unit_num)
    selected_units_embedding = tf.matmul(selected_units_embedding, key, transpose_b=True)
    selected_units_embedding = tf.cast(selected_units_embedding, tf.float32)

    autoregressive_embedding = self.dense3(autoregressive_embedding)
    autoregressive_embedding += selected_units_embedding

    return selected_units_logits, autoregressive_embedding


class ScreenLocationHead1(tf.keras.layers.Layer):
  def __init__(self):
    super(ScreenLocationHead1, self).__init__()

    self.cnn = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
    self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
    self.conv2dtranspose1 = tf.keras.layers.Conv2DTranspose(5, 4, strides=2, padding='same', activation='relu', use_bias=False)
    self.conv2dtranspose2 = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='relu', use_bias=False)
    self.dense2 = tf.keras.layers.Dense(128, activation='softmax')
    self.dense3 = tf.keras.layers.Dense(20, activation='relu')
    self.flatten = tf.keras.layers.Flatten()

  def call(self, autoregressive_embedding, map):
    batch_size = tf.shape(autoregressive_embedding)[0]

    autoregressive_embedding_reshaped = self.dense1(autoregressive_embedding)
    autoregressive_embedding_reshaped = tf.reshape(autoregressive_embedding_reshaped, [batch_size, -1,32,32])
    map_concated = tf.concat((autoregressive_embedding_reshaped, map), axis=1)

    target_location_logits = self.cnn(map_concated)
    target_location_logits = self.conv2dtranspose1(target_location_logits)
    target_location_logits = self.conv2dtranspose2(target_location_logits)
    target_location_logits = self.dense2(target_location_logits)
    target_location_logits = tf.reshape(target_location_logits, [batch_size,-1,128,128])
    target_location_logits = tf.reduce_mean(target_location_logits, axis=1)
    target_location_logits = self.flatten(target_location_logits)

    target_location = sample(target_location_logits) 
    target_location_onehot = tf.one_hot(target_location, 128*128)
    target_location_onehot = tf.cast(target_location_onehot, tf.float32) 
    target_location_embedding = self.dense3(target_location_onehot)
    autoregressive_embedding += target_location_embedding

    return target_location_logits, autoregressive_embedding


class ScreenLocationHead2(tf.keras.layers.Layer):
  def __init__(self):
    super(ScreenLocationHead2, self).__init__()

    self.cnn = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
    self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
    self.conv2dtranspose1 = tf.keras.layers.Conv2DTranspose(5, 4, strides=2, padding='same', activation='relu', use_bias=False)
    self.conv2dtranspose2 = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='relu', use_bias=False)
    self.dense2 = tf.keras.layers.Dense(128, activation='softmax')
    self.flatten = tf.keras.layers.Flatten()

  def call(self, autoregressive_embedding, map):
    batch_size = tf.shape(autoregressive_embedding)[0]

    autoregressive_embedding = self.dense1(autoregressive_embedding)
    autoregressive_embedding_reshaped = tf.reshape(autoregressive_embedding, [batch_size,-1,32,32])
    map_concated = tf.concat((autoregressive_embedding_reshaped, map), axis=1)

    target_location_logits = self.cnn(map_concated)
    target_location_logits = self.conv2dtranspose1(target_location_logits)
    target_location_logits = self.conv2dtranspose2(target_location_logits)
    target_location_logits = self.dense2(target_location_logits)
    target_location_logits = tf.reshape(target_location_logits, [batch_size,-1,128,128])
    target_location_logits = tf.reduce_mean(target_location_logits, axis=1)
    target_location_logits = self.flatten(target_location_logits)

    return target_location_logits


class Baseline(tf.keras.layers.Layer):
  def __init__(self):
    super(Baseline, self).__init__()

    self.dense = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')

  def call(self, baseline):
    value = self.dense(baseline)

    return value


def OurModel():
    lr = 0.000025

    feature_screen = tf.keras.Input(shape=[64,64,3])
    embedded_feature_units = tf.keras.Input(shape=[20,317])

    map_skip, embedded_spatial = SpatialEncoder(img_height=128, img_width=128, channel=3)(feature_screen)
    embedded_entity, entity_embeddings = EntityEncoder(317, 8)(embedded_feature_units)
    lstm_output = Core(128)(embedded_entity, embedded_spatial)
    value = Baseline()(lstm_output)
    action_type_logits, autoregressive_embedding = ActionTypeHead(6)(lstm_output)
    selected_units_logits, autoregressive_embedding = SelectedUnitsHead(20)(autoregressive_embedding,
                                                                                    action_type_logits, 
                                                                                    entity_embeddings)
    screen_target_location1_logits, autoregressive_embedding = ScreenLocationHead1()(autoregressive_embedding, map_skip)
    screen_target_location2_logits = ScreenLocationHead2()(autoregressive_embedding, map_skip)

    Actor = tf.keras.Model(
          inputs=[feature_screen, embedded_feature_units],
          outputs=[action_type_logits, selected_units_logits, screen_target_location1_logits, screen_target_location2_logits]
	)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = tf.keras.Model(
		  inputs=[feature_screen, embedded_feature_units],
		  outputs=value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic


def check_nonzero(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  #mask = unit_type
  y, x = mask.nonzero()
  indexs_nonzero_list = list(zip(x, y))
  for indexs_nonzero in indexs_nonzero_list:
    #print("indexs_nonzero:" + str(indexs_nonzero))
    x = indexs_nonzero[0]
    y = indexs_nonzero[1]
    print("mask[y,x]:" + str(mask[y,x]))


def preprocess_feature_screen(feature_screen):
  player_relative = feature_screen.player_relative / 4
  unit_hit_points_ratio = feature_screen.unit_hit_points_ratio / 255
  unit_type = feature_screen.unit_type / 200
  feature_screen_preprocessed = np.array([player_relative, unit_hit_points_ratio, unit_type])
  return feature_screen_preprocessed


class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        players = [sc2_env.Agent(sc2_env.Race['terran'])]

        feature_screen_size = 128
        feature_minimap_size = 32
        rgb_screen_size = None
        rgb_minimap_size = None
        action_space = None
        use_feature_units = True
        use_raw_units = False
        step_mul = 2
        game_steps_per_episode = None
        disable_fog = True
        visualize = False

        self.env = sc2_env.SC2Env(
              map_name=env_name,
              players=players,
              agent_interface_format=sc2_env.parse_agent_interface_format(
                  feature_screen=feature_screen_size,
                  feature_minimap=feature_minimap_size,
                  rgb_screen=rgb_screen_size,
                  rgb_minimap=rgb_minimap_size,
                  action_space=action_space,
                  use_feature_units=use_feature_units,
                  use_raw_units=use_raw_units),
              step_mul=step_mul,
              game_steps_per_episode=game_steps_per_episode,
              disable_fog=disable_fog,
              visualize=visualize)

        #self.action_size = self.env.action_space.n
        self.EPISODES, self.max_average = 10000, 5.0 # specific for pong
        self.lr = 0.000025
        self.epsilon = 0.1

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate games and plot memory
        self.state_list, self.action_list, self.reward_list = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = OurModel()

    def remember(self, state, action, reward):
        # store episode actions to memory
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)

    def act(self, feature_screen_array, entity_obs_array):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict([feature_screen_array, entity_obs_array])

        return prediction

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype=float)
        #print("reward: " + str(reward))
        for i in reversed(range(0, len(reward))):
            #if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
            #    running_add = 0

            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        try:
            discounted_r -= np.mean(discounted_r) # normalizing the result
            discounted_r /= np.std(discounted_r) # divide by standard deviation
            #print("discounted_r: " + str(discounted_r))
        except:
        	pass

        return discounted_r

    def replay(self):
        # reshape memory to appropriate shape for training
        feature_screen_list = []
        entity_obs_list = []
        for i in range(0, len(self.state_list)):
            feature_screen = self.state_list[i][0][3]['feature_screen']
            feature_screen = preprocess_feature_screen(feature_screen)
            feature_screen = np.transpose(feature_screen, (1, 2, 0))
            feature_screen_array = np.array([feature_screen])

            feature_units = self.state_list[i][0][3]['feature_units']
            entity_obs = get_entity_obs(feature_units)
            entity_obs_array = np.array([entity_obs])

            feature_screen_list.append(feature_screen_array)
            entity_obs_list.append(entity_obs_array)

        action_type_logits_list = []
        selected_units_logits_list = []
        screen_target_location1_logits_list = []
        screen_target_location2_logits_list = []
        for i in range(0, len(self.action_list)):
            action_type_logits = self.action_list[i][0]
            selected_units_logits = self.action_list[i][1]
            screen_target_location1_logits = self.action_list[i][2] 
            screen_target_location2_logits = self.action_list[i][3] 

            action_type_logits_list.append(action_type_logits)
            selected_units_logits_list.append(selected_units_logits)
            screen_target_location1_logits_list.append(screen_target_location1_logits)
            screen_target_location2_logits_list.append(screen_target_location2_logits)

        feature_screen_list_array = np.vstack(feature_screen_list)
        entity_obs_list_array = np.vstack(entity_obs_list)
        action_type_logits_list_array = np.vstack(action_type_logits_list)
        selected_units_logits_list_array = np.vstack(selected_units_logits_list)
        screen_target_location1_logits_list_array = np.vstack(screen_target_location1_logits_list)
        screen_target_location2_logits_list_array = np.vstack(screen_target_location2_logits_list)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.reward_list)
        #print("discounted_r: " + str(discounted_r))

        # Get Critic network predictions
        values = self.Critic.predict([feature_screen_list_array, entity_obs_list_array])[:, 0]
        #print("values: " + str(values))

        # Compute advantages
        discounted_r = np.reshape(discounted_r, (len(discounted_r), 1))
        advantages = discounted_r - values

        #print("feature_screen_list_array:" + str(feature_screen_list_array))
        #print("entity_obs_list_array:" + str(entity_obs_list_array))

        # training Actor and Critic networks
        self.Actor.fit(x=[feature_screen_list_array, entity_obs_list_array],
        				 y=[action_type_logits_list_array, selected_units_logits_list_array,
                            screen_target_location1_logits_list_array, screen_target_location2_logits_list_array], 
        				 sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(x=[feature_screen_list_array, entity_obs_list_array], y=discounted_r, epochs=1, verbose=0)

        # reset training memory
        self.state_list, self.action_list, self.reward_list = [], [], []
    
    def load(self):
    	#self.Actor.load('/media/kimbring2/Steam/Relational_DRL_New/Models')
    	self.Actor = load_model('/media/kimbring2/Steam/Relational_DRL_New/Models/model_1880')

    def save(self, episode):
        save_name = "/media/kimbring2/Steam/Relational_DRL_New/Models/model_{}".format(episode)
        self.Actor.save(save_name)

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + ".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def reset(self):
        frame = self.env.reset()
        state = frame
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage(next_state)
        return next_state, reward, done, info
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.reset()
            done, score, SAVING = False, 0, ''
            while not done:
                # Actor picks an action
                #print("state:" + str(state))
                feature_screen = state[0][3]['feature_screen']
                feature_screen = preprocess_feature_screen(feature_screen)
                #print("feature_screen.shape:" + str(feature_screen.shape))

                feature_screen = np.transpose(feature_screen, (1, 2, 0))
                feature_screen_array = np.array([feature_screen])

                available_actions = state[0][3]['available_actions']

                feature_units = state[0][3]['feature_units']
                
                entity_obs = get_entity_obs(feature_units)
                entity_obs_array = np.array([entity_obs])
                action_prediction = self.act(feature_screen_array, entity_obs_array)

                action_type_logits = action_prediction[0]
                selected_units_logits = action_prediction[1]
                screen_target_location1_logits = action_prediction[2] 
                screen_target_location2_logits = action_prediction[3] 
                action = [get_action(action_type_logits, selected_units_logits, 
                                       screen_target_location1_logits, screen_target_location2_logits, feature_units, self.epsilon)]

                if action[0][0] in available_actions:
                    #print("available_actions: " + str(available_actions))
                    #print("action: " + str(action))
                    next_state = self.env.step(action)
                else:
                	next_state = self.env.step([actions.FUNCTIONS.no_op()])

                done = next_state[0][0]
                if done == StepType.LAST:
                    done = True
                else:
                    done = False

                reward = next_state[0][1]

                # Memorize (state, action, reward) for training
                self.remember(state, action_prediction, reward)

                # Update current state
                state = next_state
                score += reward
                if done:
                    self.epsilon *= 0.998

                    #self.reset()
                    average = self.PlotModel(score, e)
                    #print("self.EPISODES: " + str(self.EPISODES))

                    # saving best models
                    #if average >= self.max_average:
                    if e % 20 == 0:
                        self.max_average = average
                        self.save(e)
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    
                    try:
                        self.replay()
                    except e:
                        print("self.replay() error")
                        continue
                    
                    print("episode: {}/{}, score: {}, average: {:.2f}, epsilon: {:.2f} {}".format(e, self.EPISODES, score, average, self.epsilon, SAVING))

        # close environemnt when finish training
        self.env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name)
        for e in range(100):
            state = self.reset()
            done = False
            score = 0
            while not done:
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

        self.env.close()


if __name__ == "__main__":
    env_name = 'DefeatZerglingsAndBanelings'
    agent = A2CAgent(env_name)
    #agent.load()
    agent.run()
    #agent.test('DefeatZerglingsAndBanelings_A2C_2.5e-05_Actor.h5', '')
    #agent.test('PongDeterministic-v4_A2C_1e-05_Actor.h5', '')