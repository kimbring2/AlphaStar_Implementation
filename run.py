from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import os
import abc
import sys
import math
import statistics
import random
import gym
import gc
import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler

tfd = tfp.distributions

from sklearn import preprocessing

import cv2
import threading
from threading import Thread, Lock
import time

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_NUM_FUNCTIONS = len(actions.FUNCTIONS)

is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, img_height, img_width, channel):
    super(SpatialEncoder, self).__init__()

    self.img_height = img_height
    self.img_width = img_width

    self.map_model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(13, 1, padding='same', activation=None),
       tf.keras.layers.Conv2D(16, 5, padding='same', activation=None),
       tf.keras.layers.Conv2D(32, 3, padding='same', activation=None),
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


class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number):
    super(Core, self).__init__()

    self.unit_number = unit_number
    self.dense = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_number': self.unit_number
    })
    return config

  def call(self, embedded_spatial, training=False):
    batch_size = tf.shape(embedded_spatial)[0]

    core_input = embedded_spatial
    core_input = self.dense(core_input)
    core_input = tf.reshape(core_input, (batch_size, -1, 64))

    core_output = core_input

    return core_output


class Baseline(tf.keras.layers.Layer):
  def __init__(self, action_num):
    super(Baseline, self).__init__()

    self.action_num = action_num
    self.dense = tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'action_num': self.action_num
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]

    lstm_output = tf.reshape(core_output, [batch_size,128])
    baseline = core_output
    value = self.dense(baseline)

    return value


class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, action_num):
    super(ActionTypeHead, self).__init__()

    self.action_num = action_num
    self.dense = tf.keras.layers.Dense(self.action_num, name="ActionTypeHead_dense", kernel_initializer=initializer)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'action_num': self.action_num
    })
    return config

  def call(self, lstm_output):
    batch_size = tf.shape(lstm_output)[0]
    lstm_output = tf.reshape(lstm_output, (batch_size, 2, 64))
    lstm_output_flatten = Flatten()(lstm_output)

    action_type_logits = self.dense(lstm_output_flatten)
    action_type = tf.random.categorical(action_type_logits, 1)

    return action_type_logits, action_type



class OurModel(tf.keras.Model):
  def __init__(self):
    super(OurModel, self).__init__()

    #self.screen_encoder = tf.keras.Sequential([
    #   tf.keras.layers.Conv2D(13, 1, padding='same', activation=None),
    #   tf.keras.layers.Conv2D(16, 5, padding='same', activation=None),
    #  tf.keras.layers.Conv2D(32, 3, padding='same', activation=None),
    #])

    # State Encoder
    self.spatial_encoder = SpatialEncoder(img_height=64, img_width=64, channel=3)

    # Core
    self.core = tf.keras.layers.Dense(256, activation='relu')

    # Action Head
    self.action_type_head = tf.keras.layers.Dense(_NUM_FUNCTIONS, activation='softmax')

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

    self.flatten2 = Flatten()
    self.baseline = tf.keras.layers.Dense(1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out': self.args_out
    })
    return config

  def call(self, feature_screen, feature_player):
    batch_size = tf.shape(feature_screen)[0]

    feature_player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_player, 1), 2),
                                         tf.stack([1, 32, 32, 1]))
    feature_player_encoded = tf.cast(feature_player_encoded, 'float32')

    #feature_screen_array.shape:  (1, 32, 32, 13)
    #feature_player_array.shape:  (1, 32, 32, 11)
    feature_screen_encoded = self.spatial_encoder(feature_screen)

    feature_encoded = tf.concat([feature_screen_encoded, feature_player_encoded], axis=3)
    feature_decoded_flatten = self.flatten2(feature_encoded)
    core_output = self.core(feature_decoded_flatten)

    action_type = self.action_type_head(core_output)
    
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

    value = self.baseline(core_output)

    return action_type, args_out, value


def check_nonzero(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  indexs_nonzero_list = list(zip(x, y))
  for indexs_nonzero in indexs_nonzero_list:
    x = indexs_nonzero[0]
    y = indexs_nonzero[1]


_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)

  return np.concatenate(layers, axis=0)


def preprocess_player_backup(player):
  layers = []
  for i in range(0, len(player)):
    layers.append(np.ones((32,32,1)) * np.log(player[i] + 1.0))

  return np.concatenate(layers, axis=2)


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])
FLAT_FEATURES = [
  FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
  FlatFeature(1,  features.FeatureType.SCALAR, 1, 'minerals'),
  FlatFeature(2,  features.FeatureType.SCALAR, 1, 'vespene'),
  FlatFeature(3,  features.FeatureType.SCALAR, 1, 'food_used'),
  FlatFeature(4,  features.FeatureType.SCALAR, 1, 'food_cap'),
  FlatFeature(5,  features.FeatureType.SCALAR, 1, 'food_army'),
  FlatFeature(6,  features.FeatureType.SCALAR, 1, 'food_workers'),
  FlatFeature(7,  features.FeatureType.SCALAR, 1, 'idle_worker_count'),
  FlatFeature(8,  features.FeatureType.SCALAR, 1, 'army_count'),
  FlatFeature(9,  features.FeatureType.SCALAR, 1, 'warp_gate_count'),
  FlatFeature(10, features.FeatureType.SCALAR, 1, 'larva_count'),
]
def preprocess_player(player):
  layers = []
  for s in FLAT_FEATURES:
    out = player[s.index] / s.scale
    layers.append(out)

  return layers


def preprocess_available_actions(available_action):
    available_actions = np.zeros(_NUM_FUNCTIONS, dtype=np.float32)
    available_actions[available_action] = 1

    return available_actions


def take_vector_elements(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


def actions_to_pysc2(fn_id, arg_ids, size):
  height, width = size
  actions_list = []

  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    a_l = []
    for arg_type in FUNCTIONS._func_list[a_0].args:
      arg_id = arg_ids[arg_type][n]
      if is_spatial_action[arg_type]:
        arg = [arg_id % width, arg_id // height]
      else:
        arg = [arg_id]

      a_l.append(arg)

    action = FunctionCall(a_0, a_l)
    actions_list.append(action)

  return actions_list


def mask_unused_argument_samples(fn_id, arg_ids):
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
    for arg_type in unused_types:

      #print("arg_type: ", arg_type)
      arg_ids[arg_type][n] = -1

  return fn_id, arg_ids


def mask_unavailable_actions(available_actions, fn_pi):
  fn_pi *= available_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)

  return fn_pi


def sample_actions(available_actions, fn_pi, arg_pis):
  def sample(probs):
    dist = tfd.Categorical(probs=probs)
    return dist.sample()

  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_samples = sample(fn_pi).numpy()

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi).numpy()

  return fn_samples, arg_samples


def compute_policy_entropy(available_actions, fn_pi, arg_pis, fn_id, arg_ids):
  def compute_entropy(probs):
    return -tf.reduce_sum(safe_log(probs) * probs, axis=-1)

  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  entropy = tf.reduce_mean(compute_entropy(fn_pi))
  for index, arg_type in enumerate(actions.TYPES):
    arg_id = arg_ids[index]
    arg_pi = arg_pis[arg_type]
    batch_mask = tf.cast(tf.not_equal(arg_id, -1), 'float32')
    arg_entropy = safe_div(
        tf.reduce_sum(compute_entropy(arg_pi) * batch_mask),
        tf.reduce_sum(batch_mask))
    entropy += arg_entropy

  return entropy


mse_loss = tf.keras.losses.MeanSquaredError()


feature_screen_size = 32
feature_minimap_size = 32
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
use_raw_units = False
step_mul = 8
game_steps_per_episode = None
disable_fog = False
visualize = False
class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        players = [sc2_env.Agent(sc2_env.Race['terran'])]
        self.env = sc2_env.SC2Env(
              map_name=env_name,
              players=players,
              agent_interface_format=sc2_env.parse_agent_interface_format(
                  feature_screen=feature_screen_size,
                  feature_minimap=feature_minimap_size,
                  rgb_screen=rgb_screen_size,
                  rgb_minimap=rgb_minimap_size,
                  action_space=action_space,
                  use_feature_units=use_feature_units),
              step_mul=step_mul,
              game_steps_per_episode=game_steps_per_episode,
              disable_fog=disable_fog,
              visualize=visualize)

        self.EPISODES, self.episode, self.max_average = 20000, 0, 50.0 # specific for pong
        self.lock = Lock()

        # Instantiate games and plot memory
        self.state_list, self.action_list, self.reward_list = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.workspace_path = "/media/kimbring2/Steam/Relational_DRL_New"
        self.Save_Path = 'Models'
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C'.format(self.env_name)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.ActorCritic = OurModel()
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005,
                                                                                            decay_steps=10000, decay_rate=0.94)
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate, epsilon=2e-7)

        #ssself.load()

    def act(self, feature_screen_array, feature_player_array):
        # Use the network to predict the next action to take, using the model
        prediction = self.ActorCritic(feature_screen_array, feature_player_array, training=False)
        return prediction

    @tf.function
    def discount_rewards(self, reward, dones):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma * (1 - dones[i]) + reward[i]
            discounted_r[i] = running_add

        return discounted_r

    @tf.function
    def compute_log_probs(self, probs, labels):
      labels = tf.maximum(labels, 0)
      indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
      result = tf.gather_nd(probs, indices)
      result = self.safe_log(result)

      return self.safe_log(tf.gather_nd(probs, indices)) # TODO tf.log should suffice

    @tf.function
    def mask_unavailable_actions(self, available_actions, fn_pi):
      fn_pi *= available_actions
      fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)
      return fn_pi

    @tf.function
    def safe_log(self, x):
      return tf.where(
          tf.equal(x, 0),
          tf.zeros_like(x),
          tf.math.log(tf.maximum(1e-12, x)))

    @tf.function
    def replay(self, feature_screen_list, feature_player_list, available_actions_list, 
                fn_id_list, arg_ids_list, 
                rewards, dones):
        feature_screen_array = tf.concat(feature_screen_list, 0)
        feature_player_array = tf.concat(feature_player_list, 0)
        available_actions_array = tf.concat(available_actions_list, 0)
        arg_ids_array = tf.concat(arg_ids_list, 0)

        # Compute discounted rewards
        discounted_r_array = self.discount_rewards(rewards, dones)
        with tf.GradientTape() as tape:
          prediction = self.ActorCritic(feature_screen_array, feature_player_array, training=True)
          fn_pi = prediction[0]
          arg_pis = prediction[1]
          value_estimate = prediction[2]

          discounted_r_array = tf.cast(discounted_r_array, 'float32')
          advantage = discounted_r_array - tf.stack(value_estimate)[:, 0] 

          fn_pi = self.mask_unavailable_actions(available_actions_array, fn_pi) # TODO: this should be unneccessary

          fn_log_prob = self.compute_log_probs(fn_pi, fn_id_list)
          log_prob = fn_log_prob
          for index, arg_type in enumerate(actions.TYPES):
            arg_id = arg_ids_array[:,index]
            arg_pi = arg_pis[arg_type]

            arg_log_prob = self.compute_log_probs(arg_pi, arg_id)
            arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
            log_prob += arg_log_prob

          actor_loss = -tf.math.reduce_mean(log_prob * advantage) 
          actor_loss = tf.cast(actor_loss, 'float32')

          critic_loss = mse_loss(tf.stack(value_estimate)[:, 0] , discounted_r_array)
          critic_loss = tf.cast(critic_loss, 'float32')
        
          total_loss = actor_loss + critic_loss * 0.5

        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 50.0)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

    def load(self):
        self.ActorCritic.load_weights(self.workspace_path + '/Models/model')

    def save(self):
        self.ActorCritic.save_weights(self.workspace_path + '/Models/model')

    def PlotModel(self, score, episode):
        fig = plt.figure(figsize=(18,9))
        ax = fig.add_subplot(111)
        ax.locator_params(numticks=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        #if str(episode)[-2:] == "0": # much faster than episode % 100
        if True:
            #new_list = range(0, 500)
            #plt.xticks(new_list)
            ax.plot(self.episodes, self.scores, 'b')
            ax.plot(self.episodes, self.average, 'r')
            ax.set_ylabel('Score', fontsize=18)
            ax.set_xlabel('Steps', fontsize=18)
            try:
                fig.savefig(self.path + ".png")
                plt.close(fig)
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def reset(self, env):
        frame = env.reset()
        state = frame
        return state

    def step(self, action, env):
        next_state = env.step(action)
        return next_state
    
    def train(self, n_threads):
        self.env.close()
        # Instantiate one environment per thread
        self.env_name = env_name       
        players = [sc2_env.Agent(sc2_env.Race['terran'])]

        envs = [sc2_env.SC2Env(
              map_name=env_name,
              players=players,
              agent_interface_format=sc2_env.parse_agent_interface_format(
                  feature_screen=feature_screen_size,
                  feature_minimap=feature_minimap_size,
                  rgb_screen=rgb_screen_size,
                  rgb_minimap=rgb_minimap_size,
                  action_space=action_space,
                  use_feature_units=use_feature_units),
              step_mul=step_mul,
              game_steps_per_episode=game_steps_per_episode,
              disable_fog=disable_fog,
              visualize=visualize) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self, envs[i], i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
            
        for t in threads:
            time.sleep(10)
            t.join()
        
    def train_threading(self, agent, env, thread):
        score_list = []
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.reset(env)

            feature_screen_list, feature_player_list, available_actions_list = [], [], []
            fn_id_list, arg_ids_list, rewards, dones = [], [], [], []
            while not done:
                feature_screen = state[0][3]['feature_screen']
                feature_screen = preprocess_screen(feature_screen)
                feature_screen = np.transpose(feature_screen, (1, 2, 0))

                feature_player = state[0][3]['player']
                feature_player = preprocess_player(feature_player)

                available_actions = state[0][3]['available_actions']
                available_actions = preprocess_available_actions(available_actions)

                feature_screen_array = np.array([feature_screen])
                feature_player_array = np.array([feature_player])

                feature_screen_list.append(feature_screen_array)
                feature_player_list.append(feature_player_array)
                available_actions_list.append([available_actions])

                prediction = agent.act(feature_screen_array, feature_player_array)
                fn_pi = prediction[0]
                arg_pis = prediction[1]
                value_estimate = prediction[2]

                fn_samples, arg_samples = sample_actions(available_actions, fn_pi, arg_pis)
                fn_id, arg_ids = mask_unused_argument_samples(fn_samples, arg_samples)
                fn_id_list.append(fn_id[0])

                arg_id_list = []
                for arg_type in arg_ids.keys():
                    arg_id = arg_ids[arg_type]
                    arg_id_list.append(arg_id[0])

                arg_ids_list.append(np.array([arg_id_list]))
                actions_list = actions_to_pysc2(fn_id, arg_ids, (32, 32))

                next_state = env.step(actions_list)
                done = next_state[0][0]
                if done == StepType.LAST:
                    done = True
                else:
                    done = False

                reward = float(next_state[0][1])
                rewards.append(reward)
                dones.append(done)

                score += reward
                state = next_state
                if len(feature_screen_list) == 16:
                    self.lock.acquire()
                    self.replay(feature_screen_list, feature_player_list, available_actions_list, 
                                  fn_id_list, arg_ids_list, rewards, dones)
                    self.lock.release()

                    feature_screen_list, feature_player_list, available_actions_list = [], [], []
                    fn_id_list, arg_ids_list, rewards, dones = [], [], [], []

            score_list.append(score)
            average = sum(score_list) / len(score_list)

            # Update episode count
            with self.lock:
                self.save()
                self.PlotModel(score, self.episode)
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, SAVING))
                if(self.episode < self.EPISODES):
                    self.episode += 1

        env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

        self.env.close()


if __name__ == "__main__":
    env_name = 'MoveToBeacon'
    agent = A3CAgent(env_name)
    agent.train(n_threads=1) # use as A3C
    #agent.test('Models/Pong-v0_A3C_2.5e-05_Actor.h5', '')