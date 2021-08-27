import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import sys

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
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=42, help='seed number')
parser.add_argument('--save_model', type=bool, default=None, help='save trained model')
parser.add_argument('--load_model', type=bool, default=None, help='load trained model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--gradient_clipping', type=float, default=1.0, help='gradient clipping value')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=32, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=32, help='minimap resolution')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--replay_hkl_file_path', type=str, default="replay", help='path of replay file for SL')
parser.add_argument('--sl_training', type=bool, default=False, help='Supervised Training')
parser.add_argument('--save_replay_episodes', type=int, default=10, help='minimap resolution')
parser.add_argument('--tensorboard_path', type=str, default="tensorboard", help='Folder for saving Tensorboard log file')

arguments = parser.parse_args()

if arguments.gpu_use == True:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tfd = tfp.distributions

feature_screen_size = arguments.screen_size
feature_minimap_size = arguments.minimap_size
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
use_raw_units = False
step_mul = 8
game_steps_per_episode = None
disable_fog = False
visualize = arguments.visualize

players = [sc2_env.Agent(sc2_env.Race[arguments.player_1])]

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

# Set seed for experiment reproducibility
seed = arguments.seed
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

workspace_path = arguments.workspace_path
writer = tf.summary.create_file_writer(workspace_path + "/tensorboard")

model = network.make_model(arguments.model_name)

if arguments.load_model != None:
  print("load_model")
  model.load_weights(workspace_path + 'Models/' + arguments.load_model)

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


def env_step(fn_sample: np.ndarray, screen_arg_sample: np.ndarray, minimap_arg_sample: np.ndarray, screen2_arg_sample: np.ndarray, 
	     queued_arg_sample: np.ndarray, control_group_act_arg_sample: np.ndarray, control_group_id_arg_sample: np.ndarray, 
	     select_point_act_arg_sample: np.ndarray, select_add_arg_sample: np.ndarray, select_unit_act_arg_sample: np.ndarray,
	     select_unit_id_arg_sample: np.ndarray, select_worker_arg_sample: np.ndarray, build_queue_id_arg_sample: np.ndarray,
	     unload_id_arg_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
	     np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
	     np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  args_sample = dict()
  for arg_type in actions.TYPES:
    if arg_type.name == 'screen':
      args_sample[arg_type] = screen_arg_sample
    elif arg_type.name == 'minimap':
      args_sample[arg_type] = minimap_arg_sample
    elif arg_type.name == 'screen2':
      args_sample[arg_type] = screen2_arg_sample
    elif arg_type.name == 'queued':
      args_sample[arg_type] = queued_arg_sample
    elif arg_type.name == 'control_group_act':
      args_sample[arg_type] = control_group_act_arg_sample
    elif arg_type.name == 'control_group_id':
      args_sample[arg_type] = control_group_id_arg_sample
    elif arg_type.name == 'select_point_act':
      args_sample[arg_type] =select_point_act_arg_sample
    elif arg_type.name == 'select_add':
      args_sample[arg_type] = select_add_arg_sample
    elif arg_type.name == 'select_unit_act':
      args_sample[arg_type] = select_unit_act_arg_sample
    elif arg_type.name == 'select_unit_id':
      args_sample[arg_type] = select_unit_id_arg_sample
    elif arg_type.name == 'select_worker':
      args_sample[arg_type] = select_worker_arg_sample
    elif arg_type.name == 'build_queue_id':
      args_sample[arg_type] = build_queue_id_arg_sample
    elif arg_type.name == 'unload_id':
      args_sample[arg_type] = unload_id_arg_sample

  fn_id, args_id = mask_unused_argument_samples(fn_sample, args_sample)

  arg_id_list = []
  for arg_type in actions.TYPES:
    arg_id_list.append(args_id[arg_type])

  actions_list = actions_to_pysc2(fn_id, args_id, (32, 32))
  actions_list = [actions_list]

  next_state = env.step(actions_list)
  next_state = next_state[0]
  done = next_state[0]
  if done == StepType.LAST:
    done = True
  else:
    done = False
  
  reward = float(next_state[1])
  
  feature_screen = next_state[3]['feature_screen']
  feature_screen = utils.preprocess_screen(feature_screen)
  feature_screen = np.transpose(feature_screen, (1, 2, 0))
  
  feature_minimap = next_state[3]['feature_minimap']
  feature_minimap = utils.preprocess_minimap(feature_minimap)
  feature_minimap = np.transpose(feature_minimap, (1, 2, 0))
    
  player = next_state[3]['player']
  player = utils.preprocess_player(player)
    
  available_actions = next_state[3]['available_actions']
  available_actions = utils.preprocess_available_actions(available_actions)

  feature_units = next_state[3]['feature_units']
  feature_units = utils.preprocess_feature_units(feature_units, 32)
    
  game_loop = next_state[3]['game_loop']
  
  return (feature_screen.astype(np.float32), feature_minimap.astype(np.float32), player.astype(np.float32), 
          feature_units.astype(np.float32), game_loop.astype(np.int32), available_actions.astype(np.int32), 
          np.array(reward, np.float32), np.array(done, np.float32), np.array(fn_id, np.int32),
          np.array(arg_id_list[0], np.int32), np.array(arg_id_list[1], np.int32), np.array(arg_id_list[2], np.int32), 
          np.array(arg_id_list[3], np.int32), np.array(arg_id_list[4], np.int32), np.array(arg_id_list[5], np.int32), 
          np.array(arg_id_list[6], np.int32), np.array(arg_id_list[7], np.int32), np.array(arg_id_list[8], np.int32), 
          np.array(arg_id_list[9], np.int32), np.array(arg_id_list[10], np.int32), np.array(arg_id_list[11], np.int32),
          np.array(arg_id_list[12], np.int32)
         )


def tf_env_step(fn_id: tf.Tensor, screen_arg_id: tf.Tensor, minimap_arg_id: tf.Tensor, screen2_arg_id: tf.Tensor, 
	        queued_arg_id: tf.Tensor, control_group_act_arg_id: tf.Tensor, control_group_id_arg_id: tf.Tensor, 
	        select_point_act_arg_id: tf.Tensor, select_add_arg_id: tf.Tensor, select_unit_act_arg_id: tf.Tensor, 
	        select_unit_id_arg_id: tf.Tensor, select_worker_arg_id: tf.Tensor, build_queue_id_arg_id: tf.Tensor,
	        unload_id_arg_id: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [fn_id, screen_arg_id, minimap_arg_id, screen2_arg_id, queued_arg_id, control_group_act_arg_id,
  			    control_group_id_arg_id, select_point_act_arg_id, select_add_arg_id, select_unit_act_arg_id, 
  			    select_unit_id_arg_id, select_worker_arg_id, build_queue_id_arg_id, unload_id_arg_id], 
                           [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32,
                            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
                            tf.int32, tf.int32, tf.int32, tf.int32])


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


def run_episode(
    initial_feature_screen: tf.Tensor,
    initial_feature_minimap: tf.Tensor,
    initial_players: tf.Tensor,
    initial_feature_units: tf.Tensor,
    initial_game_loop: tf.Tensor,
    initial_available_actions: tf.Tensor,  
    initial_memory_state: tf.Tensor, 
    initial_carry_state: tf.Tensor, 
    initial_done: tf.Tensor, 
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
    			      tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
    			      tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
    			      tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
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
  
  args_sample_array = tf.TensorArray(dtype=tf.int32, size=14)
  
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  
  initial_feature_screen_shape = initial_feature_screen.shape
  initial_feature_minimap_shape = initial_feature_minimap.shape
  initial_players_shape = initial_players.shape
  initial_feature_units_shape = initial_feature_units.shape
  initial_game_loop_shape = initial_game_loop.shape
  initial_available_actions_shape = initial_available_actions.shape
  initial_memory_state_shape = initial_memory_state.shape
  initial_carry_state_shape = initial_carry_state.shape
  initial_done_shape = initial_done.shape
  
  feature_screen = initial_feature_screen
  feature_minimap = initial_feature_minimap
  players = initial_players
  feature_units = initial_feature_units
  game_loop = initial_game_loop
  available_actions = initial_available_actions
  memory_state = initial_memory_state
  carry_state = initial_carry_state
  done = initial_done
  
  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    feature_screen = tf.expand_dims(feature_screen, 0)
    feature_minimap = tf.expand_dims(feature_minimap, 0)
    feature_units = tf.expand_dims(feature_units, 0)

    # Run the model and to get action probabilities and critic value
    input_ = {'feature_screen': feature_screen, 'player': players, 'feature_units': feature_units, 
              'memory_state': memory_state, 'carry_state': carry_state, 'game_loop': game_loop,
              'available_actions': tf.expand_dims(available_actions, 0)}

    prediction = model(input_, training=True)
    fn_pi = prediction['fn_out']
    args_pi = prediction['args_out']
    value = prediction['value']
    memory_state = prediction['final_memory_state']
    carry_state = prediction['final_carry_state']

    fn_pi = mask_unavailable_actions(available_actions, fn_pi)
    fn_sample = sample(fn_pi)[0]
    
    args_sample = dict()
    for arg_type, arg_pi in args_pi.items():
      arg_sample = sample(arg_pi)[0]
      args_sample_array = args_sample_array.write(arg_type.id, arg_sample)
      args_sample[arg_type] = arg_sample

    # Store critic values
    values = values.write(t, tf.squeeze(value))
    
    step_result = tf_env_step(fn_sample, args_sample_array.read(0), args_sample_array.read(1), args_sample_array.read(2), 
			       args_sample_array.read(3), args_sample_array.read(4), args_sample_array.read(5), args_sample_array.read(6),
			       args_sample_array.read(7), args_sample_array.read(8), args_sample_array.read(9), args_sample_array.read(10),
			       args_sample_array.read(11), args_sample_array.read(12))
    feature_screen = step_result[0]
    feature_minimap = step_result[1]
    player = step_result[2]
    feature_units = step_result[3] 
    game_loop = step_result[4]
    available_actions = step_result[5] 
    reward = step_result[6]
    done = step_result[7]
    fn_id = step_result[8]
    screen_arg_id = step_result[9]
    minimap_arg_id = step_result[10]
    screen2_arg_id = step_result[11]
    queued_arg_id = step_result[12]
    control_group_act_arg_id = step_result[13]
    control_group_id_arg_id = step_result[14]
    select_point_act_arg_id = step_result[15]
    select_add_arg_id = step_result[16]
    select_unit_act_arg_id = step_result[17]
    select_unit_id_arg_id = step_result[18]
    select_worker_arg_id = step_result[19]
    build_queue_id_arg_id = step_result[20]
    unload_id_arg_id = step_result[21]
    
    fn_ids = fn_ids.write(t, fn_id)
    screen_arg_ids = screen_arg_ids.write(t, screen_arg_id)
    minimap_arg_ids = minimap_arg_ids.write(t, minimap_arg_id)
    screen2_arg_ids = screen2_arg_ids.write(t, screen2_arg_id)
    queued_arg_ids = queued_arg_ids.write(t, queued_arg_id)
    control_group_act_ids = control_group_act_ids.write(t, control_group_act_arg_id)
    control_group_id_arg_ids = control_group_id_arg_ids.write(t, control_group_id_arg_id)
    select_point_act_ids = select_point_act_ids.write(t, select_point_act_arg_id)
    select_add_arg_ids = select_add_arg_ids.write(t, select_add_arg_id)
    select_unit_act_arg_ids = select_unit_act_arg_ids.write(t, select_unit_act_arg_id)
    select_unit_id_arg_ids = select_unit_id_arg_ids.write(t, select_unit_id_arg_id)
    select_worker_arg_ids = select_worker_arg_ids.write(t, select_worker_arg_id)
    build_queue_id_arg_ids = build_queue_id_arg_ids.write(t, build_queue_id_arg_id)
    unload_id_arg_ids = unload_id_arg_ids.write(t, unload_id_arg_id)
    
    fn_probs = fn_probs.write(t, fn_pi[0, fn_id])
    for arg_type, arg_pi in args_pi.items():
      if arg_type.name == 'screen':
        screen_arg_probs = screen_arg_probs.write(t, args_pi[arg_type][0, screen_arg_id])
      elif arg_type.name == 'minimap':
        minimap_arg_probs = minimap_arg_probs.write(t, args_pi[arg_type][0, minimap_arg_id])
      elif arg_type.name == 'screen2':
        screen2_arg_probs = screen2_arg_probs.write(t, args_pi[arg_type][0, screen2_arg_id])
      elif arg_type.name == 'queued':
        queued_arg_probs = queued_arg_probs.write(t, args_pi[arg_type][0, queued_arg_id])
      elif arg_type.name == 'control_group_act':
        control_group_act_probs = control_group_act_probs.write(t, args_pi[arg_type][0, control_group_act_arg_id])
      elif arg_type.name == 'control_group_id':
        control_group_id_arg_probs = control_group_id_arg_probs.write(t, args_pi[arg_type][0, control_group_id_arg_id])
      elif arg_type.name == 'select_point_act':
        select_point_act_probs = select_point_act_probs.write(t, args_pi[arg_type][0, select_point_act_arg_id])
      elif arg_type.name == 'select_add':
        select_add_arg_probs = select_add_arg_probs.write(t, args_pi[arg_type][0, select_add_arg_id])
      elif arg_type.name == 'select_unit_act':
        select_unit_act_arg_probs = select_unit_act_arg_probs.write(t, args_pi[arg_type][0, select_unit_act_arg_id])
      elif arg_type.name == 'select_unit_id':
        select_unit_id_arg_probs = select_unit_id_arg_probs.write(t, args_pi[arg_type][0, select_unit_id_arg_id])
      elif arg_type.name == 'select_worker':
        select_worker_arg_probs = select_worker_arg_probs.write(t, args_pi[arg_type][0, select_worker_arg_id])
      elif arg_type.name == 'build_queue_id':
        build_queue_id_arg_probs = build_queue_id_arg_probs.write(t, args_pi[arg_type][0, build_queue_id_arg_id])
      elif arg_type.name == 'unload_id':
        unload_id_arg_probs = unload_id_arg_probs.write(t, args_pi[arg_type][0, unload_id_arg_id])
    
    feature_screen.set_shape(initial_feature_screen_shape)
    feature_minimap.set_shape(initial_feature_minimap_shape)
    players.set_shape(initial_players_shape)
    feature_units.set_shape(initial_feature_units_shape)
    game_loop.set_shape(initial_game_loop_shape)
    available_actions.set_shape(initial_available_actions_shape)
    memory_state.set_shape(initial_memory_state_shape)
    carry_state.set_shape(initial_carry_state_shape)
    
    # Store reward
    rewards = rewards.write(t, reward)
    dones = dones.write(t, done)
    
    done.set_shape(initial_done_shape)
    if tf.cast(done, tf.bool):
      break

  last_input_ = {'feature_screen': tf.expand_dims(feature_screen, 0), 'player': players, 
                 'feature_units': tf.expand_dims(feature_units, 0), 'memory_state': memory_state, 
                 'carry_state': carry_state, 'game_loop': game_loop, 'available_actions': tf.expand_dims(available_actions, 0)}

  last_prediction = model(last_input_, training=False)
  next_value = tf.squeeze(last_prediction['value'])
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
  
  values = values.stack()
  rewards = rewards.stack()
  dones = dones.stack()
  
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
          feature_screen, feature_minimap, players, feature_units, game_loop, available_actions, memory_state, carry_state, 
          values, rewards, done, 
          fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, control_group_id_arg_ids,
          select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, select_unit_id_arg_ids, select_worker_arg_ids, 
          build_queue_id_arg_ids, unload_id_arg_ids, next_value, dones
         )
  

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
    
  returns = returns.stack()[::-1]

  #if standardize:
  #  returns = ((returns - tf.math.reduce_mean(returns)) / 
  #             (tf.math.reduce_std(returns) + eps))

  return returns

  
mse_loss = tf.keras.losses.MeanSquaredError()

def safe_log(self, x):
  return tf.where(tf.equal(x, 0), tf.zeros_like(x), tf.math.log(tf.maximum(1e-12, x)))


def compute_loss(
    fn_probs: tf.Tensor, screen_arg_probs: tf.Tensor, minimap_arg_probs: tf.Tensor, screen2_arg_probs: tf.Tensor, 
    queued_arg_probs: tf.Tensor, control_group_act_probs: tf.Tensor, control_group_id_arg_probs: tf.Tensor, 
    select_point_act_probs: tf.Tensor, select_add_arg_probs: tf.Tensor, select_unit_act_arg_probs: tf.Tensor, 
    select_unit_id_arg_probs: tf.Tensor, select_worker_arg_probs: tf.Tensor, build_queue_id_arg_probs: tf.Tensor,
    unload_id_arg_probs: tf.Tensor,
    values: tf.Tensor, returns: tf.Tensor,
    fn_ids: tf.Tensor, screen_arg_ids: tf.Tensor, minimap_arg_ids: tf.Tensor, screen2_arg_ids: tf.Tensor, 
    queued_arg_ids: tf.Tensor, control_group_act_ids: tf.Tensor, control_group_id_arg_ids: tf.Tensor, 
    select_point_act_ids: tf.Tensor, select_add_arg_ids: tf.Tensor, select_unit_act_arg_ids: tf.Tensor, 
    select_unit_id_arg_ids: tf.Tensor, select_worker_arg_ids: tf.Tensor, build_queue_id_arg_ids: tf.Tensor, 
    unload_id_arg_ids: tf.Tensor
    ) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values
  
  log_probs = tf.math.log(fn_probs)
  log_probs += tf.math.log(screen_arg_probs) * tf.cast(tf.not_equal(screen_arg_ids, -1), 'float32')
  log_probs += tf.math.log(minimap_arg_probs) * tf.cast(tf.not_equal(minimap_arg_ids, -1), 'float32')
  log_probs += tf.math.log(screen2_arg_probs) * tf.cast(tf.not_equal(screen2_arg_ids, -1), 'float32')
  log_probs += tf.math.log(queued_arg_probs) * tf.cast(tf.not_equal(queued_arg_ids, -1), 'float32')
  log_probs += tf.math.log(control_group_act_probs) * tf.cast(tf.not_equal(control_group_act_ids, -1), 'float32')
  log_probs += tf.math.log(control_group_id_arg_probs) * tf.cast(tf.not_equal(control_group_id_arg_ids, -1), 'float32')
  log_probs += tf.math.log(select_point_act_probs) * tf.cast(tf.not_equal(select_point_act_ids, -1), 'float32')
  log_probs += tf.math.log(select_add_arg_probs) * tf.cast(tf.not_equal(select_add_arg_ids, -1), 'float32')
  log_probs += tf.math.log(select_unit_act_arg_probs) * tf.cast(tf.not_equal(select_unit_act_arg_ids, -1), 'float32')
  log_probs += tf.math.log(select_unit_id_arg_probs) * tf.cast(tf.not_equal(select_unit_id_arg_ids, -1), 'float32')
  log_probs += tf.math.log(select_worker_arg_probs) * tf.cast(tf.not_equal(select_worker_arg_ids, -1), 'float32')
  log_probs += tf.math.log(build_queue_id_arg_probs) * tf.cast(tf.not_equal(build_queue_id_arg_ids, -1), 'float32')
  log_probs += tf.math.log(unload_id_arg_probs) * tf.cast(tf.not_equal(unload_id_arg_ids, -1), 'float32')
  
  actor_loss = -tf.math.reduce_mean(log_probs * tf.stop_gradient(advantage))
  critic_loss = mse_loss(values, returns)

  return actor_loss + critic_loss * 0.5
  

initial_learning_rate = arguments.learning_rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.94,
            staircase=True)

optimizer = tf.keras.optimizers.RMSprop(lr_schedule, rho=0.99, epsilon=1e-5)

@tf.function
def train_step(
    initial_feature_screen: tf.Tensor,
    initial_feature_minimap: tf.Tensor,
    initial_players: tf.Tensor,
    initial_feature_units: tf.Tensor,
    initial_game_loop: tf.Tensor,
    initial_available_actions: tf.Tensor,  
    initial_memory_state: tf.Tensor, 
    initial_carry_state: tf.Tensor, 
    initial_done: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    prediction = run_episode(initial_feature_screen, initial_feature_minimap, initial_players, initial_feature_units, 
    			      initial_game_loop, initial_available_actions, initial_memory_state, initial_carry_state, 
    			      initial_done, model, max_steps_per_episode) 
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
    
    feature_screen = prediction[14]
    feature_minimap = prediction[15] 
    players = prediction[16]
    feature_units = prediction[17] 
    game_loop = prediction[18]
    available_actions = prediction[19]
    
    memory_state = prediction[20]
    carry_state = prediction[21]
    
    values = prediction[22]
    rewards = prediction[23]
    done = prediction[24]

    fn_ids = prediction[25]
    screen_arg_ids = prediction[26]
    minimap_arg_ids = prediction[27] 
    screen2_arg_ids = prediction[28] 
    queued_arg_ids = prediction[29] 
    control_group_act_ids = prediction[30] 
    control_group_id_arg_ids = prediction[31]
    select_point_act_ids = prediction[32]
    select_add_arg_ids = prediction[33]
    select_unit_act_arg_ids = prediction[34] 
    select_unit_id_arg_ids = prediction[35] 
    select_worker_arg_ids = prediction[36] 
    build_queue_id_arg_ids = prediction[37] 
    unload_id_arg_ids = prediction[38]
    
    next_values = prediction[39]
    dones = prediction[40]

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)
    #returns = get_expected_return(rewards, dones, values, next_values, gamma)

    # Convert training data to appropriate TF tensor shapes
    converted_value = [
        tf.expand_dims(x, 1) for x in [fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, 
        				control_group_act_probs, control_group_id_arg_probs, select_point_act_probs, 
        				select_add_arg_probs, select_unit_act_arg_probs, select_unit_id_arg_probs, 
        				select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs, values, returns,
        				fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids,
        				control_group_id_arg_ids, select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids,
        				select_unit_id_arg_ids, select_worker_arg_ids, build_queue_id_arg_ids, unload_id_arg_ids
        				]] 
    fn_probs = converted_value[0]
    screen_arg_probs = converted_value[1] 
    minimap_arg_probs = converted_value[2] 
    screen2_arg_probs = converted_value[3] 
    queued_arg_probs = converted_value[4]
    control_group_act_probs = converted_value[5] 
    control_group_id_arg_probs = converted_value[6] 
    select_point_act_probs = converted_value[7]
    select_add_arg_probs = converted_value[8] 
    select_unit_act_arg_probs = converted_value[9] 
    select_unit_id_arg_probs = converted_value[10] 
    select_worker_arg_probs = converted_value[11] 
    build_queue_id_arg_probs = converted_value[12] 
    unload_id_arg_probs = converted_value[13]
    
    values = converted_value[14]
    returns = converted_value[15]
    
    fn_ids = converted_value[16]
    screen_arg_ids = converted_value[17]
    minimap_arg_ids = converted_value[18]
    screen2_arg_ids = converted_value[19]
    queued_arg_ids = converted_value[20]
    control_group_act_ids = converted_value[21]
    control_group_id_arg_ids = converted_value[22]
    select_point_act_ids = converted_value[23]
    select_add_arg_ids = converted_value[24]
    select_unit_act_arg_ids = converted_value[25]
    select_unit_id_arg_ids = converted_value[26]
    select_worker_arg_ids = converted_value[27]
    build_queue_id_arg_ids = converted_value[28]
    unload_id_arg_ids = converted_value[29]

    # Calculating loss values to update our network
    loss = compute_loss(fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, control_group_act_probs, 
			 control_group_id_arg_probs, select_point_act_probs, select_add_arg_probs, select_unit_act_arg_probs, 
			 select_unit_id_arg_probs, select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs, 
			 values, returns,
			 fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, 
			 control_group_id_arg_ids, select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, 
			 select_unit_id_arg_ids, select_worker_arg_ids, build_queue_id_arg_ids, unload_id_arg_ids
			)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)
  grads, _ = tf.clip_by_global_norm(grads, arguments.gradient_clipping)

  if arguments.training:
    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return (episode_reward, feature_screen, feature_minimap, players, feature_units, game_loop, available_actions, memory_state, 
   	  carry_state, done)
  

min_episodes_criterion = 100
max_episodes = 10000000
max_steps_per_episode = 8

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

initial_state = env.reset()
initial_state = initial_state[0]

feature_screen = initial_state[3]['feature_screen']
feature_screen = utils.preprocess_screen(feature_screen)
feature_screen = np.transpose(feature_screen, (1, 2, 0))
    
feature_minimap = initial_state[3]['feature_minimap']
feature_minimap = utils.preprocess_minimap(feature_minimap)
feature_minimap = np.transpose(feature_minimap, (1, 2, 0))
    
player = initial_state[3]['player']
player = utils.preprocess_player(player)
    
available_actions = initial_state[3]['available_actions']
available_actions = utils.preprocess_available_actions(available_actions)

feature_units = initial_state[3]['feature_units']
feature_units = utils.preprocess_feature_units(feature_units, 32)
    
game_loop = initial_state[3]['game_loop']

memory_state = tf.zeros([1,256], dtype=tf.float32)
carry_state = tf.zeros([1,256], dtype=tf.float32)

done = 0.0

episode_reward_sum = 0
episodes_reward.append(episode_reward_sum)
with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_feature_screen = tf.constant(feature_screen, dtype=tf.float32)
    initial_feature_minimap = tf.constant(feature_minimap, dtype=tf.float32)
    initial_players = tf.constant(player, dtype=tf.float32)
    initial_feature_units = tf.constant(feature_units, dtype=tf.float32)
    initial_game_loop = tf.constant(game_loop, dtype=tf.int32)
    initial_available_actions = tf.constant(available_actions, dtype=tf.int32)
    initial_done = tf.constant(done, dtype=tf.float32)
    
    train_result = train_step(initial_feature_screen, initial_feature_minimap, initial_players, initial_feature_units,
        		       initial_game_loop, initial_available_actions, memory_state, carry_state, initial_done, model, 
        		       optimizer, gamma, max_steps_per_episode)
    episode_reward = train_result[0]
    feature_screen = train_result[1]
    feature_minimap = train_result[2]  
    players = train_result[3]
    feature_units = train_result[4]  
    game_loop = train_result[5]
    available_actions = train_result[6]
    memory_state = train_result[7]
    carry_state = train_result[8]
    done = train_result[9]
    
    if done == True:
      initial_state = env.reset()
      initial_state = initial_state[0]
      
      feature_screen = initial_state[3]['feature_screen']
      feature_screen = utils.preprocess_screen(feature_screen)
      feature_screen = np.transpose(feature_screen, (1, 2, 0))
    
      feature_minimap = initial_state[3]['feature_minimap']
      feature_minimap = utils.preprocess_minimap(feature_minimap)
      feature_minimap = np.transpose(feature_minimap, (1, 2, 0))
    
      player = initial_state[3]['player']
      player = utils.preprocess_player(player)
    
      available_actions = initial_state[3]['available_actions']
      available_actions = utils.preprocess_available_actions(available_actions)

      feature_units = initial_state[3]['feature_units']
      feature_units = utils.preprocess_feature_units(feature_units, 32)
    
      game_loop = initial_state[3]['game_loop']
      
      memory_state = tf.zeros([1,256], dtype=tf.float32)
      carry_state = tf.zeros([1,256], dtype=tf.float32)
      
      with writer.as_default():
        # other model code would go here
        tf.summary.scalar("episode_reward_sum", episode_reward_sum, step=i)
        writer.flush()
      
      episodes_reward.append(episode_reward_sum)
      episode_reward_sum = 0

    episode_reward = int(episode_reward)
    episode_reward_sum += episode_reward
    #print("episode_reward_sum: ", episode_reward_sum)
    running_reward = statistics.mean(episodes_reward)

    t.set_description(f'Episode {i}')
    t.set_postfix(reward_sum=episode_reward_sum, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    if i % 1000 == 0 and arguments.save_model != None:
      model.save_weights(workspace_path + "Models/" + env_name + "_Model")
      print("save_model")
      #pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

