from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import os
import abc
import sys
import math
import argparse
import statistics
import random
import gym
import glob
import gc
import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import functools
from multiprocessing import Pool, TimeoutError
import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler

from sklearn import preprocessing
import time

import network as network
import hickle as hkl 
import utils

from multiprocessing import Process, Queue, Event
import multiprocessing as mp
from absl import flags

FLAGS = flags.FLAGS
FLAGS(['run.py'])
'''
python3.7 run_supervised_learning.py --workspace_path /home/kimbring2/AlphaStar_Implementation/ --model_name alphastar --training True --gpu_use True --learning_rate 0.0001 --replay_hkl_file_path /media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/pysc2_dataset/ --environment Simple64 --model_name alphastar
'''

parser = argparse.ArgumentParser(description='AlphaStar implementation')
parser.add_argument('--environment', type=str, default='MoveToBeacon', help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, help='root directory for checkpoint storage')
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=123, help='seed number')
parser.add_argument('--training_episode', type=int, default=5000, help='training number')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--save', type=bool, default=False, help='save trained model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=32, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=32, help='minimap resolution')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--replay_hkl_file_path', type=str, default="replay", help='path of replay file for SL')
parser.add_argument('--save_replay_episodes', type=int, default=10, help='minimap resolution')
parser.add_argument('--tensorboard_path', type=str, default="tensorboard", help='Folder for saving Tensorboard log file')

arguments = parser.parse_args()

seed = arguments.seed
tf.random.set_seed(seed)
np.random.seed(seed)

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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

if arguments.gpu_use == True:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def check_nonzero(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  indexs_nonzero_list = list(zip(x, y))
  for indexs_nonzero in indexs_nonzero_list:
    x = indexs_nonzero[0]
    y = indexs_nonzero[1]


def take_vector_elements(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


def actions_to_pysc2(fn_id, arg_ids, size):
  height, width = size
  actions_list = []

  a_0 = int(fn_id)
  a_l = []
  for arg_type in FUNCTIONS._func_list[a_0].args:
    arg_id = int(arg_ids[arg_type])
    if is_spatial_action[arg_type]:
      arg = [arg_id % width, arg_id // height]
    else:
      arg = [arg_id]

    a_l.append(arg)

  action = FunctionCall(a_0, a_l)
  actions_list.append(action)

  return actions_list


def mask_unused_argument_samples(fn_id, arg_ids):
  args_out = dict()
  for arg_type in actions.TYPES:
    args_out[arg_type] = arg_ids[arg_type][0]

  a_0 = fn_id[0]
  unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
  for arg_type in unused_types:
    args_out[arg_type] = -1

  return fn_id, args_out


def mask_unavailable_actions(available_actions, fn_pi):
  available_actions = available_actions
  available_actions = tf.cast(available_actions, 'float32')

  fn_pi *= available_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)

  return fn_pi


@tf.function
def sample_actions(available_actions, fn_pi, arg_pis):
  def sample(probs):
    dist = tfd.Categorical(probs=probs)
    return dist.sample()

  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi)

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


env_name = arguments.environment
workspace_path = arguments.workspace_path
Save_Path = 'Models'
        
model = network.make_model(arguments.model_name)

if arguments.pretrained_model != None:
  model.load_weights(workspace_path + "/Models/" + arguments.pretrained_model)

writer = tf.summary.create_file_writer(workspace_path + "/tensorboard/supervised_learning")

feature_screen_size = arguments.screen_size
feature_minimap_size = arguments.minimap_size
class TrajetoryDataset(tf.data.Dataset):
  def _generator(num_trajectorys):
    while True:
      replay_file_path_list = glob.glob(arguments.replay_hkl_file_path + '*.hkl')
      replay_file_path = random.choice(replay_file_path_list)
      try:
        replay = hkl.load(replay_file_path)
      except:
        continue

      home_replay_done = False

      home_replay_feature_screen_list, home_replay_feature_minimap_list = [], []
      home_replay_player_list, home_replay_feature_units_list = [], []
      home_replay_available_actions_list, last_action_type_list = [], []
      home_replay_fn_id_list, home_replay_arg_ids_list = [], []
      home_replay_game_loop_list, home_replay_build_queue_list = [], []
      home_replay_single_select_list, home_replay_multi_select_list = [], []
      home_replay_score_cumulative_list = []

      last_action_type = [0]

      replay_file_length = len(replay['home_game_loop'])
      num_samples = replay_file_length
      for sample_idx in range(1, num_samples):
          home_replay_feature_screen = replay['home_feature_screen'][sample_idx-1]
          home_replay_feature_screen = utils.preprocess_screen(home_replay_feature_screen)
          home_replay_feature_screen = np.transpose(home_replay_feature_screen, (1, 2, 0))

          home_replay_feature_minimap = replay['home_feature_minimap'][sample_idx-1]
          home_replay_feature_minimap = utils.preprocess_minimap(home_replay_feature_minimap)
          home_replay_feature_minimap = np.transpose(home_replay_feature_minimap, (1, 2, 0))

          home_replay_player = replay['home_player'][sample_idx-1]
          home_replay_player = utils.preprocess_player(home_replay_player)

          home_replay_feature_units = replay['home_feature_units'][sample_idx-1]
          home_replay_feature_units = utils.preprocess_feature_units(home_replay_feature_units, feature_screen_size)

          home_replay_game_loop = replay['home_game_loop'][sample_idx-1]

          home_replay_available_actions = replay['home_available_actions'][sample_idx-1]
          home_replay_available_actions = utils.preprocess_available_actions(home_replay_available_actions)

          home_replay_build_queue = replay['home_build_queue'][sample_idx-1]
          home_replay_build_queue = utils.preprocess_build_queue(home_replay_build_queue)

          home_replay_single_select = replay['home_single_select'][sample_idx-1]
          home_replay_single_select = utils.preprocess_single_select(home_replay_single_select)

          home_replay_multi_select = replay['home_multi_select'][sample_idx-1]
          home_replay_multi_select = utils.preprocess_multi_select(home_replay_multi_select)

          home_replay_score_cumulative = replay['home_score_cumulative'][sample_idx-1]
          home_replay_score_cumulative = utils.preprocess_score_cumulative(home_replay_score_cumulative)

          home_replay_feature_screen_array = np.array([home_replay_feature_screen])
          home_replay_feature_minimap_array = np.array([home_replay_feature_minimap])
          home_replay_player_array = np.array([home_replay_player])
          home_replay_feature_units_array = np.array([home_replay_feature_units])
          home_replay_available_actions_array = np.array([home_replay_available_actions])
          home_replay_game_loop_array = np.array([home_replay_game_loop])
          last_action_type_array = np.array([last_action_type])
          home_replay_build_queue_array = np.array([home_replay_build_queue])
          home_replay_single_select_array = np.array([home_replay_single_select])
          home_replay_multi_select_array = np.array([home_replay_multi_select])
          home_replay_score_cumulative_array = np.array([home_replay_score_cumulative])

          home_replay_actions = replay['home_action'][sample_idx]
          home_replay_action = random.choice(home_replay_actions)
          home_replay_fn_id = int(home_replay_action[0])
          home_replay_feature_screen_list.append(home_replay_feature_screen_array[0])
          home_replay_feature_minimap_list.append(home_replay_feature_minimap_array[0])
          home_replay_player_list.append(home_replay_player_array[0])
          home_replay_feature_units_list.append(home_replay_feature_units_array[0])
          home_replay_available_actions_list.append(home_replay_available_actions_array[0])
          home_replay_game_loop_list.append(home_replay_game_loop_array[0])
          last_action_type_list.append(np.array([last_action_type[0]]))
          home_replay_build_queue_list.append(home_replay_build_queue_array[0])
          home_replay_single_select_list.append(home_replay_single_select_array[0])
          home_replay_multi_select_list.append(home_replay_multi_select_array[0])
          home_replay_score_cumulative_list.append(home_replay_score_cumulative_array[0])

          home_replay_args_ids = dict()
          for arg_type in actions.TYPES:
            home_replay_args_ids[arg_type] = -1

          arg_index = 0
          for arg_type in FUNCTIONS._func_list[home_replay_fn_id].args:
              home_replay_args_ids[arg_type] = home_replay_action[1][arg_index]
              arg_index += 1

          last_action_type = [home_replay_fn_id]
          home_replay_fn_id_list.append([home_replay_fn_id])
          home_replay_arg_id_list = []
          for arg_type in home_replay_args_ids.keys():
              arg_id = home_replay_args_ids[arg_type]
              if type(arg_id) == list:
                if len(arg_id) == 2:
                  arg_id = arg_id[0] + arg_id[1] * feature_screen_size
                else:
                  arg_id = int(arg_id[0])

              home_replay_arg_id_list.append(arg_id)

          home_replay_arg_ids_list.append(np.array(home_replay_arg_id_list))
          if sample_idx == replay_file_length - 1:
            home_replay_done = True

          if home_replay_done == True:
            yield (home_replay_feature_screen_list, home_replay_feature_minimap_list, 
                    home_replay_player_list, home_replay_feature_units_list, 
                    home_replay_available_actions_list,
                    home_replay_fn_id_list, home_replay_arg_ids_list,
                    home_replay_game_loop_list, last_action_type_list,
                    home_replay_build_queue_list, home_replay_single_select_list,
                    home_replay_multi_select_list, home_replay_score_cumulative_list
                    )
              
            home_replay_feature_screen_list, home_replay_feature_minimap_list = [], []
            home_replay_player_list, home_replay_feature_units_list = [], []
            home_replay_available_actions_list, last_action_type_list = [], []
            home_replay_fn_id_list, home_replay_arg_ids_list = [], []
            home_replay_game_loop_list, home_replay_build_queue_list = [], []
            home_replay_single_select_list, home_replay_multi_select_list = [], []
            home_replay_score_cumulative_list = []

            break

  def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, 
                           tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.int32, 
                           tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32,
                           tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32,
                           tf.dtypes.float32),
          args=(num_trajectorys,)
      )

dataset = tf.data.Dataset.range(1).interleave(TrajetoryDataset, 
  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)


cce = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(arguments.learning_rate)
@tf.function
def supervised_replay(replay_feature_screen_list, replay_feature_minimap_list,
                      replay_player_list, replay_feature_units_list, 
                      replay_available_actions_list, replay_fn_id_list, replay_args_ids_list,
                      memory_state, carry_state,
                      replay_game_loop_list, last_action_type_list,
                      replay_build_queue_list, replay_single_select_list, replay_multi_select_list,
                      replay_score_cumulative_list):
    replay_feature_screen_array = tf.concat(replay_feature_screen_list, 0)
    replay_feature_minimap_array = tf.concat(replay_feature_minimap_list, 0)
    replay_player_array = tf.concat(replay_player_list, 0)
    replay_feature_units_array = tf.concat(replay_feature_units_list, 0)
    replay_memory_state_array = tf.concat(memory_state, 0)
    replay_carry_state_array = tf.concat(carry_state, 0)
    replay_game_loop_array = tf.concat(replay_game_loop_list, 0)
    last_action_type_array = tf.concat(last_action_type_list, 0)
    replay_available_actions_array = tf.concat(replay_available_actions_list, 0)
    replay_fn_id_array = tf.concat(replay_fn_id_list, 0)
    replay_arg_ids_array = tf.concat(replay_args_ids_list, 0)

    replay_build_queue_array = tf.concat(replay_build_queue_list, 0)
    replay_single_select_array = tf.concat(replay_single_select_list, 0)
    replay_multi_select_array = tf.concat(replay_multi_select_list, 0)
    replay_score_cumulative_array = tf.concat(replay_score_cumulative_list, 0)

    memory_state = replay_memory_state_array
    carry_state = replay_carry_state_array

    batch_size = replay_feature_screen_array.shape[0]
    with tf.GradientTape() as tape:
      fn_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      screen_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      minimap_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      screen2_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      queued_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      control_group_act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      control_group_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      select_point_act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      select_add_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      select_unit_act_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      select_unit_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      select_worker_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      build_queue_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      unload_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      for i in tf.range(0, batch_size):
        input_dict = {'feature_screen': tf.expand_dims(replay_feature_screen_array[i,:,:,:], 0), 
                      'feature_minimap': tf.expand_dims(replay_feature_minimap_array[i,:,:,:], 0),
                      'player': tf.expand_dims(replay_player_array[i,:], 0), 
                      'feature_units': tf.expand_dims(replay_feature_units_array[i,:,:], 0), 
                      'memory_state': memory_state, 'carry_state': carry_state, 
                      'game_loop': tf.expand_dims(replay_game_loop_array[i,:], 0), 
                      'available_actions': tf.expand_dims(replay_available_actions_array[i,:], 0), 
                      'last_action_type': tf.expand_dims(last_action_type_array[i,:], 0), 
                      'build_queue': tf.expand_dims(replay_build_queue_array[i], 0), 
                      'single_select': tf.expand_dims(replay_single_select_array[i], 0), 
                      'multi_select': tf.expand_dims(replay_multi_select_array[i], 0), 
                      'score_cumulative': tf.expand_dims(replay_score_cumulative_array[i], 0)}
        prediction = model(input_dict, training=True)
        fn_pi = prediction['fn_out']
        args_pi = prediction['args_out']
        memory_state = prediction['final_memory_state']
        carry_state = prediction['final_carry_state']
        fn_probs = fn_probs.write(i, fn_pi[0])

        arg_ids_loss = 0
        for index, arg_type in enumerate(actions.TYPES):
          if arg_type.name == 'screen':
            screen_arg_probs = screen_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'minimap':
            minimap_arg_probs = minimap_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'screen2':
            screen2_arg_probs = screen2_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'queued':
            queued_arg_probs = queued_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'control_group_act':
            control_group_act_probs = control_group_act_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'control_group_id':
            control_group_id_arg_probs = control_group_id_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'select_point_act':
            select_point_act_probs = select_point_act_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'select_add':
            select_add_arg_probs = select_add_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'select_unit_act':
            select_unit_act_arg_probs = select_unit_act_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'select_unit_id':
            select_unit_id_arg_probs = select_unit_id_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'select_worker':
            select_worker_arg_probs = select_worker_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'build_queue_id':
            build_queue_id_arg_probs = build_queue_id_arg_probs.write(i, args_pi[arg_type][0])
          elif arg_type.name == 'unload_id':
            unload_id_arg_probs = unload_id_arg_probs.write(i, args_pi[arg_type][0])

      fn_probs = fn_probs.stack()
      screen_arg_probs = screen_arg_probs.stack()
      minimap_arg_probs = minimap_arg_probs.stack()
      screen2_arg_probs = screen2_arg_probs.stack()
      queued_arg_probs = queued_arg_probs.stack()
      control_group_act_probs = control_group_act_probs.stack()
      control_group_id_arg_probs = control_group_id_arg_probs.stack()
      select_point_act_probs = select_point_act_probs.stack()
      select_add_arg_probs = select_add_arg_probs.stack()
      select_unit_act_arg_probs = select_unit_act_arg_probs.stack()
      select_unit_id_arg_probs = select_unit_id_arg_probs.stack()
      select_worker_arg_probs = select_worker_arg_probs.stack()
      build_queue_id_arg_probs = build_queue_id_arg_probs.stack()
      unload_id_arg_probs = unload_id_arg_probs.stack()

      tf.print("replay_fn_id_array: ", replay_fn_id_array)
      tf.print("tf.argmax(fn_probs, 1): ", tf.argmax(fn_probs, 1))

      replay_fn_id_array_onehot = tf.one_hot(replay_fn_id_array, 573)
      replay_fn_id_array_onehot = tf.reshape(replay_fn_id_array_onehot, (batch_size, 573))
      replay_fn_id_array_onehot *= replay_available_actions_array
      fn_id_loss = cce(replay_fn_id_array_onehot, fn_probs)

      arg_ids_loss = 0 
      for index, arg_type in enumerate(actions.TYPES):
        if arg_type.name == 'screen':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, screen_arg_probs.shape[1])
          screen_arg_loss = cce(replay_arg_id_array_onehot, screen_arg_probs)
          #tf.print("screen_arg_loss: ", screen_arg_loss)
          arg_ids_loss += screen_arg_loss
        elif arg_type.name == 'minimap':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, minimap_arg_probs.shape[1])
          minimap_arg_loss = cce(replay_arg_id_array_onehot, minimap_arg_probs)
          #tf.print("minimap_arg_loss: ", minimap_arg_loss)
          arg_ids_loss += minimap_arg_loss
        elif arg_type.name == 'screen2':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, screen2_arg_probs.shape[1])
          screen2_arg_loss = cce(replay_arg_id_array_onehot, screen2_arg_probs)
          #tf.print("screen2_arg_loss: ", screen2_arg_loss)
          arg_ids_loss += screen2_arg_loss
        elif arg_type.name == 'queued':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, queued_arg_probs.shape[1])
          queued_arg_loss = cce(replay_arg_id_array_onehot, queued_arg_probs)
          #tf.print("queued_arg_loss: ", queued_arg_loss)
          arg_ids_loss += queued_arg_loss
        elif arg_type.name == 'control_group_act':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, control_group_act_probs.shape[1])
          control_group_act_loss = cce(replay_arg_id_array_onehot, control_group_act_probs)
          #tf.print("control_group_act_loss: ", control_group_act_loss)
          arg_ids_loss += control_group_act_loss
        elif arg_type.name == 'control_group_id':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, control_group_id_arg_probs.shape[1])
          control_group_id_arg_loss = cce(replay_arg_id_array_onehot, control_group_id_arg_probs)
          #tf.print("control_group_id_arg_loss: ", control_group_id_arg_loss)
          arg_ids_loss += control_group_id_arg_loss
        elif arg_type.name == 'select_point_act':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, select_point_act_probs.shape[1])
          select_point_act_loss = cce(replay_arg_id_array_onehot, select_point_act_probs)
          #tf.print("select_point_act_loss: ", select_point_act_loss)
          arg_ids_loss += select_point_act_loss
        elif arg_type.name == 'select_add':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, select_add_arg_probs.shape[1])
          select_add_arg_loss = cce(replay_arg_id_array_onehot, select_add_arg_probs)
          #tf.print("select_add_arg_loss: ", select_add_arg_loss)
          arg_ids_loss += select_add_arg_loss
        elif arg_type.name == 'select_unit_act':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, select_unit_act_arg_probs.shape[1])
          select_unit_act_arg_loss = cce(replay_arg_id_array_onehot, select_unit_act_arg_probs)
          #tf.print("select_unit_act_arg_loss: ", select_unit_act_arg_loss)
          arg_ids_loss += select_unit_act_arg_loss
        elif arg_type.name == 'select_unit_id':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, select_unit_id_arg_probs.shape[1])
          select_unit_id_arg_loss = cce(replay_arg_id_array_onehot, select_unit_id_arg_probs)
          #tf.print("select_unit_id_arg_loss: ", select_unit_id_arg_loss)
          arg_ids_loss += select_unit_id_arg_loss
        elif arg_type.name == 'select_worker':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, select_worker_arg_probs.shape[1])
          select_worker_arg_loss = cce(replay_arg_id_array_onehot, select_worker_arg_probs)
          #tf.print("select_worker_arg_loss: ", select_worker_arg_loss)
          arg_ids_loss += select_worker_arg_loss
        elif arg_type.name == 'build_queue_id':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, build_queue_id_arg_probs.shape[1])
          build_queue_id_arg_loss = cce(replay_arg_id_array_onehot, build_queue_id_arg_probs)
          #tf.print("build_queue_id_arg_loss: ", build_queue_id_arg_loss)
          arg_ids_loss += build_queue_id_arg_loss
        elif arg_type.name == 'unload_id':
          replay_arg_id = replay_arg_ids_array[:,index]
          replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, unload_id_arg_probs.shape[1])
          unload_id_arg_loss = cce(replay_arg_id_array_onehot, unload_id_arg_probs)
          #tf.print("unload_id_arg_loss: ", unload_id_arg_loss)
          arg_ids_loss += unload_id_arg_loss

      tf.print("fn_id_loss: ", fn_id_loss)
      tf.print("arg_ids_loss: ", arg_ids_loss)
      regularization_loss = tf.reduce_sum(model.losses)
      total_loss = fn_id_loss + arg_ids_loss + 1e-5 * regularization_loss
      tf.print("")

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, memory_state, carry_state


def supervised_train(dataset, training_episode):
    def iter_dataset(_dataset):
        dataset_iterator = iter(_dataset)
        while True:
            yield next(dataset_iterator)

    training_step = 0
    for batch in dataset:
      episode_size = batch[0].shape[1]

      replay_feature_screen_list = batch[0][0]
      replay_feature_minimap_list = batch[1][0]
      replay_player_list = batch[2][0]
      replay_feature_units_list = batch[3][0]
      replay_available_actions_list = batch[4][0]
      replay_fn_id_list = batch[5][0]
      replay_args_ids_list = batch[6][0]
      replay_game_loop_list = batch[7][0]
      replay_last_action_type_list = batch[8][0]
      replay_build_queue_list = batch[9][0]
      replay_single_select_list = batch[10][0]
      replay_multi_select_list = batch[11][0]
      replay_score_cumulative_list = batch[12][0]

      memory_state = np.zeros([1,1024], dtype=np.float32)
      carry_state =  np.zeros([1,1024], dtype=np.float32)
      step_length = 8
      for episode_index in range(0, episode_size, step_length):
        feature_screen = replay_feature_screen_list[episode_index:episode_index+step_length,:,:,:]
        feature_minimap = replay_feature_minimap_list[episode_index:episode_index+step_length,:,:,:]
        player = replay_player_list[episode_index:episode_index+step_length,:]
        feature_units = replay_feature_units_list[episode_index:episode_index+step_length,:,:]
        available_actions = replay_available_actions_list[episode_index:episode_index+step_length,:]
        fn_id_list = replay_fn_id_list[episode_index:episode_index+step_length,:]
        args_ids = replay_args_ids_list[episode_index:episode_index+step_length,:]
        game_loop = replay_game_loop_list[episode_index:episode_index+step_length,:]
        last_action_type = replay_last_action_type_list[episode_index:episode_index+step_length,:]
        build_queue = replay_build_queue_list[episode_index:episode_index+step_length]
        single_select = replay_single_select_list[episode_index:episode_index+step_length]
        multi_select = replay_multi_select_list[episode_index:episode_index+step_length]
        score_cumulative = replay_score_cumulative_list[episode_index:episode_index+step_length]
        if arguments.training == True and len(feature_screen) == step_length:
          total_loss, next_memory_state, next_carry_state = supervised_replay(feature_screen, feature_minimap, 
                                                                              player, feature_units, 
                                                                              available_actions, fn_id_list, args_ids,
                                                                              memory_state, carry_state, 
                                                                              game_loop, last_action_type,
                                                                              build_queue, single_select,
                                                                              multi_select, score_cumulative)
          memory_state = next_memory_state
          carry_state = next_carry_state

          training_step += 1
          print("training_step: {}".format(training_step))

          if training_step % 250 == 0:
            with writer.as_default():
              tf.summary.scalar("total_loss", total_loss, step=training_step)
              writer.flush()

          if training_step % 5000 == 0:
            model.save_weights(workspace_path + '/Models/supervised_model_' + str(training_step / 10000))


def main():
  supervised_train(dataset, arguments.training_episode)


if __name__ == "__main__":
  main()
