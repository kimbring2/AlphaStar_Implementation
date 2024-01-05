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

from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import tensorflow_probability as tfp

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

from pysc2.lib import actions, features, units
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

from absl import flags
import argparse

import network
import utils

FLAGS = flags.FLAGS
FLAGS(['run.py'])

parser = argparse.ArgumentParser(description='AlphaStar implementation')
parser.add_argument('--environment', type=str, default='MoveToBeacon', 
                    choices=["MoveToBeacon", "CollectMineralShards", "FindAndDefeatZerglings",
                             "DefeatRoaches", "DefeatZerglingsAndBanelings"],
                    help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, help='root directory for checkpoint storage')
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', 
                    choices=["fullyconv", "alphastar", "relationalfullyconv"], help='model name')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=42, help='seed number')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=16, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=16, help='minimap resolution')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--save_replay_episodes', type=int, default=10, help='minimap resolution')

arguments = parser.parse_args()

if arguments.gpu_use == True:
  physical_devices = tf.config.list_physical_devices('GPU') 
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

minigame_environment_list = ['MoveToBeacon', 'DefeatRoaches', 'BuildMarines', 'DefeatZerglingsAndBanelings']
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

#print("arguments.pretrained_model: ", arguments.pretrained_model)
if arguments.pretrained_model:
  #model.load_weights(workspace_path + "Models/" + arguments.pretrained_model)
  model.load_weights(os.path.join(workspace_path, "model", arguments.pretrained_model))


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type.name] = name in ['minimap', 'screen', 'screen2']


def actions_to_pysc2(fn_id, arg_ids, size):
  height, width = size
  actions_list = []

  a_0 = int(fn_id)
  a_l = []
  for arg_type in FUNCTIONS._func_list[a_0].args:
      arg_id = int(arg_ids[arg_type.name])
      if is_spatial_action[arg_type.name]:
          arg = [arg_id % width, arg_id // height]
      else:
          arg = [arg_id]

      a_l.append(arg)

  action = FunctionCall(a_0, a_l)
  actions_list.append(action)

  return actions_list


def mask_unavailable_actions(available_actions, fn_pi):
  available_actions = tf.cast(available_actions, 'float32')
  fn_pi *= available_actions

  return fn_pi


def sample(logits):
    dist = tfd.Categorical(logits=logits)
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

    feature_screen_history = np.zeros(((feature_screen_size, feature_screen_size, 6*4)))
    act_history = np.zeros((16, utils._NUM_FUNCTIONS))
    while True:
        state = state[0]

        feature_screen = state[3]['feature_screen']
        # feature_screen.shape:  (27, feature_screen_size, feature_screen_size)

        feature_screen = utils.preprocess_screen(feature_screen)
        feature_screen = np.transpose(feature_screen, (1, 2, 0))
        #feature_screen = np.expand_dims(feature_screen, 0)

        feature_screen_history = np.roll(feature_screen_history, 6, axis=2)
        feature_screen_history[:,:,0:6] = feature_screen
        
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

        last_actions = state[3]['last_actions']
        if len(last_actions) != 0:
          last_actions_decoded = utils.preprocess_available_actions(last_actions[0])
        else:
          last_actions_decoded = utils.preprocess_available_actions(0)

        act_history = np.roll(act_history, 1, axis=0)
        act_history[0,:] = last_actions_decoded
        #print("act_history.shape: ", act_history.shape)
        #print("")
        '''
        model_input = {'feature_screen': np.expand_dims(feature_screen_history, 0), 
                       'feature_minimap': feature_minimap, 'player': player, 'feature_units': feature_units, 
                       'memory_state': memory_state, 'carry_state': carry_state, 'game_loop': game_loop, 
                       'available_actions': available_actions, 'build_queue': build_queue, 'single_select': single_select, 
                       'multi_select': multi_select, 'score_cumulative': score_cumulative, 
                       'act_history': np.expand_dims(act_history, 0)}
        prediction = model(model_input, training=True)
        '''
        prediction = model([np.expand_dims(feature_screen_history, 0), feature_minimap, player, feature_units, game_loop, 
                            available_actions, build_queue, single_select, multi_select, score_cumulative, 
                            np.expand_dims(act_history, 0), memory_state, carry_state], training=False)

        fn_pi = prediction[0]

        fn_pi = tf.nn.softmax(fn_pi)
        fn_pi = mask_unavailable_actions(available_actions, fn_pi)
        fn_probs = fn_pi / tf.reduce_sum(fn_pi, axis=1, keepdims=True)
        fn_dist = tfd.Categorical(probs=fn_probs)
        fn_id_samples = fn_dist.sample()[0]
        fn_id = int(fn_id_samples)

        screen_arg_samples = sample(prediction[1])[0]
        minimap_arg_samples = sample(prediction[2])[0]
        screen2_arg_samples = sample(prediction[3])[0]
        queued_arg_samples = sample(prediction[4])[0]
        control_group_act_arg_samples = sample(prediction[5])[0]
        control_group_id_arg_samples = sample(prediction[6])[0]
        select_point_act_arg_samples = sample(prediction[7])[0]
        select_add_arg_samples = sample(prediction[8])[0]
        select_unit_act_arg_samples = sample(prediction[9])[0]
        select_unit_id_arg_samples = sample(prediction[10])[0]
        select_worker_arg_samples = sample(prediction[11])[0]
        build_queue_id_arg_samples = sample(prediction[12])[0]
        unload_id_arg_samples = sample(prediction[13])[0]

        args_id = dict()
        args_id['screen'] = int(screen_arg_samples)
        args_id['minimap'] = int(minimap_arg_samples)
        args_id['screen2'] = int(screen2_arg_samples)
        args_id['queued'] = int(queued_arg_samples)
        args_id['control_group_act'] = int(control_group_act_arg_samples)
        args_id['control_group_id'] = int(control_group_id_arg_samples)
        args_id['select_point_act'] = int(select_point_act_arg_samples)
        args_id['select_add'] = int(select_add_arg_samples)
        args_id['select_unit_act'] = int(select_unit_act_arg_samples)
        args_id['select_unit_id'] = int(select_unit_id_arg_samples)
        args_id['select_worker'] = int(select_worker_arg_samples)
        args_id['build_queue_id'] = int(build_queue_id_arg_samples)
        args_id['unload_id'] = int(unload_id_arg_samples)

        memory_state = prediction[15]
        carry_state = prediction[16]

        #print("fn_id: ", fn_id)
        #print("args_id: ", args_id)

        #fn_id = 0
        actions_list = actions_to_pysc2(fn_id, args_id, (feature_screen_size, feature_screen_size))
        #actions_list = [actions_list]

        next_state = env.step(actions_list)
        done = next_state[0][0]

        reward = float(next_state[0][1])
        if done == StepType.LAST:
          done = True
        else:
          done = False

        reward_sum += reward

        state = next_state

        step += 1
        if done:
            print("Score: {0}, Step: {1}".format(reward_sum, step))
            step = 0
            reward_sum = 0  

            #state = tf.constant(env.reset(), dtype=tf.float32)
            memory_state = tf.zeros([1,256], dtype=tf.float32)
            carry_state = tf.zeros([1,256], dtype=tf.float32)

            break

env.close()