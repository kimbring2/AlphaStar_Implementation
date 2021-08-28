import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import sys
import os
import glob
import random
import hickle as hkl 

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import tensorflow_probability as tfp

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

from absl import flags
import argparse

import network
import utils

FLAGS = flags.FLAGS
FLAGS(['run.py'])

parser = argparse.ArgumentParser(description='AlphaStar implementation')
parser.add_argument('--environment', type=str, default='MoveToBeacon', help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, help='root directory for checkpoint storage')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=42, help='seed number')
parser.add_argument('--load', type=bool, default=False, help='load pretrained model')
parser.add_argument('--save', type=bool, default=False, help='save trained model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=32, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=32, help='minimap resolution')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--replay_hkl_file_path', type=str, default="replay", help='path of replay file for SL')
parser.add_argument('--tensorboard_path', type=str, default="tensorboard", help='Folder for saving Tensorboard log file')

arguments = parser.parse_args()

if arguments.gpu_use == True:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tfd = tfp.distributions

feature_screen_size = arguments.screen_size
feature_minimap_size = arguments.minimap_size
'''
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
use_raw_units = False
step_mul = 4
game_steps_per_episode = None
disable_fog = False
visualize = arguments.visualize
'''
minigame_environment_list = ['MoveToBeacon', 'DefeatRoaches', 'BuildMarines']
if arguments.environment not in minigame_environment_list:
  players = [sc2_env.Agent(sc2_env.Race[arguments.player_1]), sc2_env.Bot(sc2_env.Race[arguments.player_2], sc2_env.Difficulty.very_easy)]
else:
  players = [sc2_env.Agent(sc2_env.Race[arguments.player_1])]
'''
# Create the environment
env_name = arguments.environment
env = sc2_env.SC2Env(
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
'''
# Set seed for experiment reproducibility
seed = arguments.seed
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

workspace_path = arguments.workspace_path
writer = tf.summary.create_file_writer(workspace_path + "/tensorboard/supervised_learning/")

model = network.make_model(arguments.model_name)

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
    args_out[arg_type] = arg_ids[arg_type]

  a_0 = fn_id
  unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
  for arg_type in unused_types:
    args_out[arg_type] = -1

  return fn_id, args_out

'''
def replay_step(start_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  print("batch[0][0]: ", batch[0][0])

  replay_feature_screen = batch[0][0][start_index,:,:,:]
  replay_feature_minimap = batch[1][0][start_index,:,:,:]
  replay_player = batch[2][0][start_index,:]
  replay_feature_units = batch[3][0][start_index,:,:]
  replay_available_actions = batch[4][0][start_index,:]
  replay_fn_id = batch[5][0][start_index,:]
  replay_args_id = batch[6][0][start_index,:]
  replay_game_loop = batch[7][0][start_index,:]
  replay_last_action_type = batch[8][0][start_index,:]
  replay_build_queue = batch[9][0][start_index,]
  replay_single_select = batch[10][0][start_index]
  replay_multi_select = batch[11][0][start_index]
  replay_score_cumulative = batch[12][0][start_index]

  return (replay_feature_screen.astype(np.float32), replay_feature_minimap.astype(np.float32), replay_player.astype(np.float32), 
           replay_feature_units.astype(np.float32), replay_game_loop.astype(np.int32), replay_available_actions.astype(np.int32), 
           replay_build_queue.astype(np.float32), replay_single_select.astype(np.float32),
           replay_multi_select.astype(np.float32), replay_score_cumulative.astype(np.float32),
           np.array(replay_fn_id, np.int32), np.array(replay_args_id, np.int32),
           )


def tf_replay_step(start_index: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(replay_step, [start_index], 
                                [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32,
                                 tf.int32, tf.int32])

'''


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def mask_unavailable_actions(available_actions, fn_pi):
  available_actions = tf.cast(available_actions, 'float32')
  fn_pi *= available_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)

  return fn_pi


def sample(probs):
    dist = tfd.Categorical(probs=probs)
    return dist.sample()


def run_supervised_episode(
    start_index: tf.Tensor,
    memory_state: tf.Tensor, 
    carry_state: tf.Tensor, 
    model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                          tf.Tensor]:
  """Runs a single episode to collect training data."""
  
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

  available_actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

  fn_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  screen_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  minimap_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  screen2_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  queued_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  control_group_act_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  control_group_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_point_act_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_add_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_unit_act_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_unit_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_worker_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  build_queue_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  unload_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  memory_state_shape = memory_state.shape
  carry_state_shape = carry_state.shape

  for t in range(0, step_length):
    replay_feature_screen = tf.expand_dims(batch[0][0][start_index + t,:,:,:], axis=0)
    replay_feature_minimap = tf.expand_dims(batch[1][0][start_index + t,:,:,:], axis=0)
    replay_player = tf.expand_dims(batch[2][0][start_index + t,:], axis=0)
    replay_feature_units = tf.expand_dims(batch[3][0][start_index + t,:,:], axis=0)
    replay_available_actions = tf.expand_dims(batch[4][0][start_index + t,:], axis=0)
    replay_fn_id = batch[5][0][start_index + t,:]
    replay_args_id = batch[6][0][start_index + t,:]
    replay_game_loop = tf.expand_dims(batch[7][0][start_index + t,:], axis=0)
    replay_last_action = tf.expand_dims(batch[8][0][start_index + t,:], axis=0)
    replay_build_queue = tf.expand_dims(batch[9][0][start_index + t], axis=0)
    replay_single_select = tf.expand_dims(batch[10][0][start_index + t], axis=0)
    replay_multi_select = tf.expand_dims(batch[11][0][start_index + t], axis=0)
    replay_score_cumulative = tf.expand_dims(batch[12][0][start_index + t], axis=0)

    #replay_fn_id = step_result[10]
    fn_ids = fn_ids.write(t, replay_fn_id[0])

    screen_arg_ids = screen_arg_ids.write(t, replay_args_id[0])
    minimap_arg_ids = minimap_arg_ids.write(t, replay_args_id[1])
    screen2_arg_ids = screen2_arg_ids.write(t, replay_args_id[2])
    queued_arg_ids = queued_arg_ids.write(t, replay_args_id[3])
    control_group_act_ids = control_group_act_ids.write(t, replay_args_id[4])
    control_group_id_arg_ids = control_group_id_arg_ids.write(t, replay_args_id[5])
    select_point_act_ids = select_point_act_ids.write(t, replay_args_id[6])
    select_add_arg_ids = select_add_arg_ids.write(t, replay_args_id[7])
    select_unit_act_arg_ids = select_unit_act_arg_ids.write(t, replay_args_id[8])
    select_unit_id_arg_ids = select_unit_id_arg_ids.write(t, replay_args_id[9])
    select_worker_arg_ids = select_worker_arg_ids.write(t, replay_args_id[10])
    build_queue_id_arg_ids = build_queue_id_arg_ids.write(t, replay_args_id[11])
    unload_id_arg_ids = unload_id_arg_ids.write(t, replay_args_id[12])

    available_actions = available_actions.write(t, replay_available_actions)

    model_input = {'feature_screen': replay_feature_screen, 'feature_minimap': replay_feature_minimap,
                      'player': replay_player, 'feature_units': replay_feature_units, 
                      'memory_state': memory_state, 'carry_state': carry_state, 
                      'game_loop': replay_game_loop, 'available_actions': replay_available_actions, 
                      'build_queue': replay_build_queue, 'single_select': replay_single_select, 
                      'multi_select': replay_multi_select, 
                      'score_cumulative': replay_score_cumulative}

    prediction = model(model_input, training=True)
    fn_pi = prediction['fn_out']
    args_pi = prediction['args_out']
    value = prediction['value']
    memory_state = prediction['final_memory_state']
    carry_state = prediction['final_carry_state']

    fn_probs = fn_probs.write(t, fn_pi[0])
    for arg_type in actions.TYPES:
      if arg_type.name == 'screen':
        screen_arg_probs = screen_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'minimap':
        minimap_arg_probs = minimap_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'screen2':
        screen2_arg_probs = screen2_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'queued':
        queued_arg_probs = queued_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'control_group_act':
        control_group_act_probs = control_group_act_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'control_group_id':
        control_group_id_arg_probs = control_group_id_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'select_point_act':
        select_point_act_probs = select_point_act_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'select_add':
        select_add_arg_probs = select_add_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'select_unit_act':
        select_unit_act_arg_probs = select_unit_act_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'select_unit_id':
        select_unit_id_arg_probs = select_unit_id_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'select_worker':
        select_worker_arg_probs = select_worker_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'build_queue_id':
        build_queue_id_arg_probs = build_queue_id_arg_probs.write(t, args_pi[arg_type][0])
      elif arg_type.name == 'unload_id':
        unload_id_arg_probs = unload_id_arg_probs.write(t, args_pi[arg_type][0])

    memory_state.set_shape(memory_state_shape)
    carry_state.set_shape(carry_state_shape)
  
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

  available_actions = available_actions.stack()

  fn_ids = fn_ids.stack()
  screen_arg_ids = screen_arg_ids.stack()
  minimap_arg_ids = minimap_arg_ids.stack()
  screen2_arg_ids = screen2_arg_ids.stack()
  queued_arg_ids = queued_arg_ids.stack()
  control_group_act_ids = control_group_act_ids.stack()
  control_group_id_arg_ids = control_group_id_arg_ids.stack()
  select_point_act_ids = select_point_act_ids.stack()
  select_add_arg_ids = select_add_arg_ids.stack()
  select_unit_act_arg_ids = select_unit_act_arg_ids.stack()
  select_unit_id_arg_ids = select_unit_id_arg_ids.stack()
  select_worker_arg_ids = select_worker_arg_ids.stack()
  build_queue_id_arg_ids = build_queue_id_arg_ids.stack()
  unload_id_arg_ids = unload_id_arg_ids.stack()

  return (fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, control_group_act_probs, 
            control_group_id_arg_probs, select_point_act_probs, select_add_arg_probs, select_unit_act_arg_probs, 
            select_unit_id_arg_probs, select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs,
            available_actions, memory_state, carry_state, 
            fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, control_group_id_arg_ids,
            select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, select_unit_id_arg_ids, select_worker_arg_ids, 
            build_queue_id_arg_ids, unload_id_arg_ids
           )


mse_loss = tf.keras.losses.MeanSquaredError()
cce_loss = tf.keras.losses.CategoricalCrossentropy()
batch_size = 8

@tf.function
def compute_supervised_loss(
    fn_probs: tf.Tensor, screen_arg_probs: tf.Tensor, minimap_arg_probs: tf.Tensor, screen2_arg_probs: tf.Tensor, 
    queued_arg_probs: tf.Tensor, control_group_act_probs: tf.Tensor, control_group_id_arg_probs: tf.Tensor, 
    select_point_act_probs: tf.Tensor, select_add_arg_probs: tf.Tensor, select_unit_act_arg_probs: tf.Tensor, 
    select_unit_id_arg_probs: tf.Tensor, select_worker_arg_probs: tf.Tensor, build_queue_id_arg_probs: tf.Tensor,
    unload_id_arg_probs: tf.Tensor,
    available_actions: tf.Tensor,
    fn_ids: tf.Tensor, screen_arg_ids: tf.Tensor, minimap_arg_ids: tf.Tensor, screen2_arg_ids: tf.Tensor, 
    queued_arg_ids: tf.Tensor, control_group_act_ids: tf.Tensor, control_group_id_arg_ids: tf.Tensor, 
    select_point_act_ids: tf.Tensor, select_add_arg_ids: tf.Tensor, select_unit_act_arg_ids: tf.Tensor, 
    select_unit_id_arg_ids: tf.Tensor, select_worker_arg_ids: tf.Tensor, build_queue_id_arg_ids: tf.Tensor, 
    unload_id_arg_ids: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""
  fn_ids_array_onehot = tf.one_hot(fn_ids, 573)

  available_actions = tf.squeeze(available_actions, 1)
  available_actions = tf.cast(available_actions, 'float32')
  fn_ids_array_onehot *= available_actions

  fn_id_loss = cce_loss(fn_ids_array_onehot, fn_probs)

  screen_arg_ids_onehot = tf.one_hot(screen_arg_ids, screen_arg_probs.shape[1])
  minimap_arg_ids_onehot = tf.one_hot(minimap_arg_ids, minimap_arg_probs.shape[1])
  screen2_arg_ids_onehot = tf.one_hot(screen2_arg_ids, screen2_arg_probs.shape[1])
  queued_arg_ids_onehot = tf.one_hot(queued_arg_ids, queued_arg_probs.shape[1])
  control_group_act_ids_onehot = tf.one_hot(control_group_act_ids, control_group_act_probs.shape[1])
  control_group_id_arg_ids_onehot = tf.one_hot(control_group_id_arg_ids, control_group_id_arg_probs.shape[1])
  select_point_act_ids_onehot = tf.one_hot(select_point_act_ids, select_point_act_probs.shape[1])
  select_add_arg_ids_onehot = tf.one_hot(select_add_arg_ids, select_add_arg_probs.shape[1])
  select_unit_act_arg_ids_onehot = tf.one_hot(select_unit_act_arg_ids, select_unit_act_arg_probs.shape[1])
  select_unit_id_arg_ids_onehot = tf.one_hot(select_unit_id_arg_ids, select_unit_id_arg_probs.shape[1])
  select_worker_arg_ids_onehot = tf.one_hot(select_worker_arg_ids, select_worker_arg_probs.shape[1])
  build_queue_id_arg_ids_onehot = tf.one_hot(build_queue_id_arg_ids, build_queue_id_arg_probs.shape[1])
  unload_id_arg_ids_onehot = tf.one_hot(unload_id_arg_ids, unload_id_arg_probs.shape[1])

  arg_ids_loss = 0 
  arg_ids_loss += cce_loss(screen_arg_ids_onehot, screen_arg_probs)
  arg_ids_loss += cce_loss(minimap_arg_ids_onehot, minimap_arg_probs)
  arg_ids_loss += cce_loss(screen2_arg_ids_onehot, screen2_arg_probs)
  arg_ids_loss += cce_loss(queued_arg_ids_onehot, queued_arg_probs)
  arg_ids_loss += cce_loss(control_group_act_ids_onehot, control_group_act_probs)
  arg_ids_loss += cce_loss(control_group_id_arg_ids_onehot, control_group_id_arg_probs)
  arg_ids_loss += cce_loss(select_point_act_ids_onehot, select_point_act_probs)
  arg_ids_loss += cce_loss(select_add_arg_ids_onehot, select_add_arg_probs)
  arg_ids_loss += cce_loss(select_unit_act_arg_ids_onehot, select_unit_act_arg_probs)
  arg_ids_loss += cce_loss(select_unit_id_arg_ids_onehot, select_unit_id_arg_probs)
  arg_ids_loss += cce_loss(select_worker_arg_ids_onehot, select_worker_arg_probs)
  arg_ids_loss += cce_loss(build_queue_id_arg_ids_onehot, build_queue_id_arg_probs)
  arg_ids_loss += cce_loss(unload_id_arg_ids_onehot, unload_id_arg_probs)

  regularization_loss = tf.reduce_sum(model.losses)

  total_loss = fn_id_loss + arg_ids_loss + 1e-5 * regularization_loss

  return total_loss

def train_supervised_step(
    start_index: tf.Tensor,
    memory_state: tf.Tensor, 
    carry_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    prediction = run_supervised_episode(start_index, memory_state, carry_state, model) 

    fn_probs = prediction[0] 
    screen_arg_probs = prediction[1]  
    minimap_arg_probs = prediction[2] 
    screen2_arg_probs = prediction[3] 
    queued_arg_probs = prediction[4] 
    control_group_act_probs = prediction[5] 
    control_group_id_arg_probs = prediction[6] 
    select_point_act_probs = prediction[7] 
    select_add_arg_probs = prediction[8] 
    select_unit_act_arg_probs = prediction[9] 
    select_unit_id_arg_probs = prediction[10]
    select_worker_arg_probs = prediction[11] 
    build_queue_id_arg_probs = prediction[12] 
    unload_id_arg_probs = prediction[13]

    available_actions = prediction[14]

    memory_state = prediction[15]
    carry_state = prediction[16]

    fn_ids = prediction[17]
    screen_arg_ids = prediction[18]
    minimap_arg_ids = prediction[19] 
    screen2_arg_ids = prediction[20] 
    queued_arg_ids = prediction[21] 
    control_group_act_ids = prediction[22] 
    control_group_id_arg_ids = prediction[23]
    select_point_act_ids = prediction[24]
    select_add_arg_ids = prediction[25]
    select_unit_act_arg_ids = prediction[26] 
    select_unit_id_arg_ids = prediction[27] 
    select_worker_arg_ids = prediction[28] 
    build_queue_id_arg_ids = prediction[29] 
    unload_id_arg_ids = prediction[30]

    # Calculating loss values to update our network
    loss = compute_supervised_loss(fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, 
                                          control_group_act_probs, control_group_id_arg_probs, select_point_act_probs, 
                                          select_add_arg_probs, select_unit_act_arg_probs, select_unit_id_arg_probs, 
                                          select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs, 
                                          available_actions,
                                          fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, 
                                          control_group_id_arg_ids, select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, 
                                          select_unit_id_arg_ids, select_worker_arg_ids, build_queue_id_arg_ids, unload_id_arg_ids)
    

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss, memory_state, carry_state


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
      home_replay_feature_player_list, home_replay_feature_units_list = [], []
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

          home_replay_feature_player = replay['home_player'][sample_idx-1]
          home_replay_feature_player = utils.preprocess_player(home_replay_feature_player)

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
          home_replay_feature_player_array = np.array([home_replay_feature_player])
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
          home_replay_feature_player_list.append(home_replay_feature_player_array[0])
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
                    home_replay_feature_player_list, home_replay_feature_units_list, 
                    home_replay_available_actions_list,
                    home_replay_fn_id_list, home_replay_arg_ids_list,
                    home_replay_game_loop_list, last_action_type_list,
                    home_replay_build_queue_list, home_replay_single_select_list,
                    home_replay_multi_select_list, home_replay_score_cumulative_list
                    )
              
            home_replay_feature_screen_list, home_replay_feature_minimap_list = [], []
            home_replay_feature_player_list, home_replay_feature_units_list = [], []
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

supervised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

step_length = 8
steps = 0
for batch in dataset:
  start_index = 0 
  episode_size = batch[0].shape[1]
  memory_state = tf.zeros([1,256], dtype=tf.float32)
  carry_state = tf.zeros([1,256], dtype=tf.float32)
  while True:
    loss, memory_state, carry_state = train_supervised_step(start_index, memory_state, carry_state, model, supervised_optimizer)
    start_index += 8
    steps += 1
            
    if start_index + step_length >= episode_size:
      start_index = 0
      memory_state = tf.zeros([1,256], dtype=tf.float32)
      carry_state = tf.zeros([1,256], dtype=tf.float32)

    if steps % 1000 == 0:
      model.save_weights(workspace_path + "Models/Supervised_Learning/" + env_name + "_Model_" + str(steps))

    if steps % 100 == 0:
      with writer.as_default():
        # other model code would go here
        tf.summary.scalar("loss", loss, step=steps)
        writer.flush()