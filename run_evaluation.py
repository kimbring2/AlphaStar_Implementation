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
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=42, help='seed number')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=32, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=32, help='minimap resolution')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--save_replay_episodes', type=int, default=10, help='minimap resolution')

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
step_mul = 4
game_steps_per_episode = None
disable_fog = False
visualize = arguments.visualize

minigame_environment_list = ['MoveToBeacon', 'DefeatRoaches', 'BuildMarines']
if arguments.environment not in minigame_environment_list:
  players = [sc2_env.Agent(sc2_env.Race[arguments.player_1]), sc2_env.Bot(sc2_env.Race[arguments.player_2], sc2_env.Difficulty.very_easy)]
else:
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

model = network.make_model(arguments.model_name)
#model.load_weights(workspace_path + "/Models/Supervised_Learning/" + env_name + "_Model")
model.load_weights(workspace_path + "/Models/" + "supervised_model_10.0")

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


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def mask_unavailable_actions(available_actions, fn_pi):
  available_actions = tf.cast(available_actions, 'float32')
  fn_pi *= available_actions
  #print("fn_pi b:" , fn_pi)
  #print("tf.reduce_sum(fn_pi, axis=1, keepdims=True)[0][0]:" , tf.reduce_sum(fn_pi, axis=1, keepdims=True)[0][0])
  #if tf.reduce_sum(fn_pi, axis=1, keepdims=True)[0][0] != 0.0:
  #  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)[0][0]
    #print("fn_pi a:" , fn_pi)

  return fn_pi


def sample(probs):
    dist = tfd.Categorical(probs=probs)
    return dist.sample()


def mask_unused_argument_samples(fn_id, arg_ids):
  args_out = dict()
  for arg_type in actions.TYPES:
    args_out[arg_type] = arg_ids[arg_type]

  a_0 = fn_id
  unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
  for arg_type in unused_types:
    args_out[arg_type] = -1

  return fn_id, args_out


for i_episode in range(0, 10000):
    state = env.reset()
    memory_state = tf.zeros([1,256], dtype=tf.float32)
    carry_state = tf.zeros([1,256], dtype=tf.float32)

    step = 0
    reward_sum = 0
    while True:
        state = state[0]
        feature_screen = state[3]['feature_screen']
        feature_screen = utils.preprocess_screen(feature_screen)
        feature_screen = np.transpose(feature_screen, (1, 2, 0))
        feature_screen = np.expand_dims(feature_screen, 0)
        
        feature_minimap = state[3]['feature_minimap']
        feature_minimap = utils.preprocess_minimap(feature_minimap)
        feature_minimap = np.transpose(feature_minimap, (1, 2, 0))
        feature_minimap = np.expand_dims(feature_minimap, 0)
        
        player = state[3]['player']
        player = utils.preprocess_player(player)
        player = np.expand_dims(player, 0)
        
        available_actions = state[3]['available_actions']
        available_actions = utils.preprocess_available_actions(available_actions)
        available_actions = np.expand_dims(available_actions, 0)

        feature_units = state[3]['feature_units']
        feature_units = utils.preprocess_feature_units(feature_units, 32)
        feature_units = np.expand_dims(feature_units, 0)

        game_loop = state[3]['game_loop']
        game_loop = np.expand_dims(game_loop, 0)

        build_queue = state[3]['build_queue']
        build_queue = utils.preprocess_build_queue(build_queue)
        build_queue = np.expand_dims(build_queue, 0)

        single_select = state[3]['single_select']
        single_select = utils.preprocess_single_select(single_select)
        single_select = np.expand_dims(single_select, 0)

        multi_select = state[3]['multi_select']
        multi_select = utils.preprocess_multi_select(multi_select)
        multi_select = np.expand_dims(multi_select, 0)

        score_cumulative = state[3]['score_cumulative']
        score_cumulative = utils.preprocess_score_cumulative(score_cumulative)
        score_cumulative = np.expand_dims(score_cumulative, 0)

        model_input = {'feature_screen': feature_screen, 'feature_minimap': feature_minimap,
                          'player': player, 'feature_units': feature_units, 
                          'memory_state': memory_state, 'carry_state': carry_state, 
                          'game_loop': game_loop, 'available_actions': available_actions, 
                          'build_queue': build_queue, 'single_select': single_select, 
                          'multi_select': multi_select, 
                          'score_cumulative': score_cumulative}

        prediction = model(model_input, training=True)
        fn_pi = prediction['fn_out']
        args_pi = prediction['args_out']
        next_memory_state = prediction['final_memory_state']
        next_carry_state = prediction['final_carry_state']
        #print("fn_pi b:" , fn_pi)
        fn_pi = mask_unavailable_actions(available_actions, fn_pi)
        #print("fn_pi a:" , fn_pi)
        fn_sample = sample(fn_pi)[0]
        
        args_sample = dict()
        for arg_type, arg_pi in args_pi.items():
          arg_sample = sample(arg_pi)[0]
          args_sample[arg_type] = arg_sample

        #print("fn_sample:" , fn_sample)
        fn_id, args_id = mask_unused_argument_samples(fn_sample, args_sample)

        actions_list = actions_to_pysc2(fn_id, args_id, (32, 32))
        actions_list = [actions_list]

        next_state = env.step(actions_list)
        done = next_state[0][0]

        reward = float(next_state[0][1])
        if done == StepType.LAST:
          done = True
        else:
          done = False

        state = next_state
        memory_state = next_memory_state
        carry_state = next_carry_state

        step += 1
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, total_step))
            step = 0
            reward_sum = 0  

            state = tf.constant(env.reset(), dtype=tf.float32)
            memory_state = tf.zeros([1,256], dtype=tf.float32)
            carry_state = tf.zeros([1,256], dtype=tf.float32)

            break

env.close()