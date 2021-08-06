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

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler

from sklearn import preprocessing
import cv2
import time

import network as network
import agent as agent
import trajectory as trajectory
import hickle as hkl 
import utils

from absl import flags

FLAGS = flags.FLAGS
FLAGS(['run.py'])

parser = argparse.ArgumentParser(description='AlphaStar implementation')
parser.add_argument('--environment', type=str, default='MoveToBeacon', help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, help='root directory for checkpoint storage')
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=123, help='seed number')
parser.add_argument('--load', type=bool, default=False, help='load pretrained model')
parser.add_argument('--save', type=bool, default=False, help='save trained model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--gradient_clipping', type=float, default=50.0, help='gradient clipping value')
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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
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
        
if not os.path.exists(Save_Path): os.makedirs(Save_Path)
path = '{}_A2C_{}_{}_{}'.format(env_name, seed, arguments.learning_rate, arguments.gradient_clipping)
scores, episodes, average = [], [], []
def PlotModel(score, episode):
    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(111)
    ax.locator_params(numticks=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    scores.append(score)
    episodes.append(episode)
    average.append(sum(scores[-50:]) / len(scores[-50:]))
    if True:
        ax.plot(episodes, scores, 'b')
        ax.plot(episodes, average, 'r')
        ax.set_ylabel('Score', fontsize=18)
        ax.set_xlabel('Steps', fontsize=18)
        try:
            fig.savefig(path +  ".png")
            plt.close(fig)
        except OSError:
            pass

    return average[-1]

#home_agent = agent.A2CAgent(network.ConvLSTM(screen_size=arguments.screen_size, minimap_size=arguments.minimap_size), 
#                               arguments.learning_rate, 
#                               arguments.gradient_clipping)
model = network.make_model('alphastar')
home_agent = agent.A2CAgent(model, arguments.learning_rate, arguments.gradient_clipping)
#agent_2 = agent.A2CAgent(network.OurModel())
writer = tf.summary.create_file_writer(arguments.tensorboard_path)


def supervised_train(training_episode):
    # Initialization
    #home_agent.load(workspace_path + '/Models/supervised_model')

    for episode in range(0, training_episode):
        home_agent.save(workspace_path + '/Models/supervised_model')

        replay_file_path_list = glob.glob(arguments.replay_hkl_file_path + '*.*')
        replay_file_path = random.choice(replay_file_path_list)

        replay = hkl.load(replay_file_path)
        replay_file_length = len(replay['home_game_loop'])

        replay_index = 0
        home_replay_done = False

        home_replay_feature_screen_list, home_replay_feature_minimap_list = [], []
        home_replay_feature_player_list, home_replay_feature_units_list = [], []
        home_replay_available_actions_list, last_action_type_list = [], []
        home_replay_fn_id_list, home_replay_arg_ids_list = [], []
        home_replay_memory_state_list, home_replay_carry_state_list = [], []
        replay_game_loop_list = []

        memory_state = np.zeros([1,256], dtype=np.float32)
        carry_state = np.zeros([1,256], dtype=np.float32)
        last_action_type = [0]
        for replay_index in range(0, replay_file_length):
            home_replay_feature_screen = replay['home_feature_screen'][replay_index]
            home_replay_feature_screen = utils.preprocess_screen(home_replay_feature_screen)
            home_replay_feature_screen = np.transpose(home_replay_feature_screen, (1, 2, 0))

            home_replay_feature_minimap = replay['home_feature_minimap'][replay_index]
            home_replay_feature_minimap = utils.preprocess_minimap(home_replay_feature_minimap)
            home_replay_feature_minimap = np.transpose(home_replay_feature_minimap, (1, 2, 0))
            #print("home_replay_feature_minimap.shape: ", home_replay_feature_minimap.shape)

            home_replay_feature_player = replay['home_player'][replay_index]
            home_replay_feature_player = utils.preprocess_player(home_replay_feature_player)

            home_replay_feature_units = replay['home_feature_units'][replay_index]
            home_replay_feature_units = utils.preprocess_feature_units(home_replay_feature_units, feature_screen_size)

            replay_game_loop = replay['home_game_loop'][replay_index] / 15000.0

            home_replay_available_actions = replay['home_available_actions'][replay_index]
            home_replay_available_actions = utils.preprocess_available_actions(home_replay_available_actions)

            home_replay_feature_screen_array = np.array([home_replay_feature_screen])
            home_replay_feature_minimap_array = np.array([home_replay_feature_minimap])
            home_replay_feature_player_array = np.array([home_replay_feature_player])
            home_replay_feature_units_array = np.array([home_replay_feature_units])
            home_replay_available_actions_array = np.array([home_replay_available_actions])
            replay_game_loop_array = np.array([replay_game_loop])
            last_action_type_array = np.array([last_action_type])

            home_replay_prediction = home_agent.act(home_replay_feature_screen_array, home_replay_feature_minimap_array,
                                                             home_replay_feature_player_array, home_replay_feature_units_array, 
                                                             home_replay_available_actions_array, memory_state, carry_state,
                                                             replay_game_loop_array, last_action_type_array)
            home_replay_next_memory_state = home_replay_prediction['final_memory_state']
            home_replay_next_carry_state = home_replay_prediction['final_carry_state']

            home_replay_actions = replay['home_action'][replay_index]
            home_replay_action = random.choice(home_replay_actions)
            home_replay_fn_id = int(home_replay_action[0])

            home_replay_feature_screen_list.append(home_replay_feature_screen_array)
            home_replay_feature_minimap_list.append(home_replay_feature_minimap_array)
            home_replay_feature_player_list.append(home_replay_feature_player_array)
            home_replay_feature_units_list.append(home_replay_feature_units_array)
            home_replay_available_actions_list.append(home_replay_available_actions_array)
            home_replay_memory_state_list.append(memory_state)
            home_replay_carry_state_list.append(carry_state)
            replay_game_loop_list.append(replay_game_loop_array)
            last_action_type_list.append(np.array([last_action_type]))

            home_replay_args_ids = dict()
            for arg_type in actions.TYPES:
              home_replay_args_ids[arg_type] = -1

            arg_index = 0
            for arg_type in FUNCTIONS._func_list[home_replay_fn_id].args:
                home_replay_args_ids[arg_type] = home_replay_action[1][arg_index]
                arg_index += 1

            last_action_type = [home_replay_fn_id]
            home_replay_fn_id_list.append(home_replay_fn_id)
            home_replay_arg_id_list = []
            for arg_type in home_replay_args_ids.keys():
                arg_id = home_replay_args_ids[arg_type]
                if type(arg_id) == list:
                  if len(arg_id) == 2:
                    arg_id = arg_id[0] + arg_id[1] * feature_screen_size
                  else:
                    arg_id = int(arg_id[0])

                home_replay_arg_id_list.append(arg_id)

            home_replay_arg_ids_list.append(np.array([home_replay_arg_id_list]))
            if replay_index == replay_file_length - 1:
              home_replay_done = True
            
            memory_state = home_replay_next_memory_state
            carry_state =  home_replay_next_carry_state

            if len(home_replay_feature_screen_list) != 0 and len(home_replay_feature_screen_list) == 16 or home_replay_done == True:
                print("episode: ", episode)

                if arguments.training == True:
                  total_loss = home_agent.supervised_replay(home_replay_feature_screen_list, home_replay_feature_minimap_list,
                                                                      home_replay_feature_player_list, home_replay_feature_units_list, 
                                                                      home_replay_available_actions_list, home_replay_fn_id_list, 
                                                                      home_replay_arg_ids_list,
                                                                      home_replay_memory_state_list, home_replay_carry_state_list,
                                                                      replay_game_loop_list, last_action_type_list)

                home_replay_feature_screen_list, home_replay_feature_minimap_list = [], []
                home_replay_feature_player_list, home_replay_feature_units_list= [], []
                home_replay_available_actions_list, last_action_type_list = [], []
                home_replay_fn_id_list, home_replay_arg_ids_list = [], []
                home_replay_memory_state_list, home_replay_carry_state_list = [], []
                replay_game_loop_list = []

                with writer.as_default():
                  tf.summary.scalar("total_loss", total_loss, step=episode)
                  writer.flush()

                if home_replay_done:
                    break


feature_screen_size = arguments.screen_size
feature_minimap_size = arguments.minimap_size
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
use_raw_units = False
step_mul = 2
game_steps_per_episode = None
disable_fog = False

minigame_environment_list = ['MoveToBeacon', 'DefeatRoaches', 'BuildMarines']

if arguments.environment not in minigame_environment_list:
  players = [sc2_env.Agent(sc2_env.Race[arguments.player_1]), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
else:
  players = [sc2_env.Agent(sc2_env.Race[arguments.player_1]), sc2_env.Agent(sc2_env.Race[arguments.player_1])]

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
      visualize=arguments.visualize,
      replay_dir=arguments.replay_dir,
      save_replay_episodes=arguments.save_replay_episodes)


def reinforcement_train(training_episode):
    score_list = []
    EPISODES, episode, max_average, SAVING  = 20000, 0, 5.0, ''
  
    if arguments.load != False:
        #home_agent.load(workspace_path + '/Models/BuildMarines/reinforcment_model')
        home_agent.load(workspace_path + '/Models/supervised_model')

    while episode < training_episode:
        # Reset episode
        home_score, home_done, SAVING = 0, False, ''
        opponent_score, opponent_done = 0, False
        state = env.reset()

        home_feature_screen_list, home_feature_minimap_list = [], []
        home_feature_player_list, home_feature_units_list = [], []
        home_available_actions_list, last_action_type_list = [], []
        home_fn_id_list, home_arg_ids_list, home_rewards, home_dones = [], [], [], []
        home_memory_state_list, home_carry_state_list = [], []
        game_loop_list, home_feature_screen_history_list = [], []

        memory_state = np.zeros([1,256], dtype=np.float32)
        carry_state =  np.zeros([1,256], dtype=np.float32)

        home_state = state[0]
        home_feature_screen = home_state[3]['feature_screen']
        home_feature_screen = utils.preprocess_screen(home_feature_screen)
        home_feature_screen = np.transpose(home_feature_screen, (1, 2, 0))
        home_feature_screens_t = np.stack((home_feature_screen, home_feature_screen, home_feature_screen, home_feature_screen), axis=0)

        delay = 0
        last_action_type = [0]
        while not home_done:
            home_state = state[0]
            game_loop = home_state[3]['game_loop'] / 15000.0

            home_feature_screen = home_state[3]['feature_screen']
            home_feature_screen = utils.preprocess_screen(home_feature_screen)
            home_feature_screen = np.transpose(home_feature_screen, (1, 2, 0))

            home_feature_player = home_state[3]['player']
            home_feature_player = utils.preprocess_player(home_feature_player)

            home_feature_minimap = home_state[3]['feature_minimap']
            home_feature_minimap = utils.preprocess_minimap(home_feature_minimap)
            home_feature_minimap = np.transpose(home_feature_minimap, (1, 2, 0))

            home_available_actions = home_state[3]['available_actions']
            home_available_actions = utils.preprocess_available_actions(home_available_actions)

            home_feature_units = home_state[3]['feature_units']
            home_feature_units = utils.preprocess_feature_units(home_feature_units, feature_screen_size)

            home_feature_screen_array = np.array([home_feature_screen])
            home_feature_minimap_array = np.array([home_feature_minimap])
            home_feature_player_array = np.array([home_feature_player])
            home_feature_units_array = np.array([home_feature_units])
            home_available_actions_array = np.array([home_available_actions])
            game_loop_array = np.array([game_loop])
            last_action_type_array = np.array([last_action_type])
            home_feature_screen_history_array = np.array([home_feature_screens_t])

            home_feature_screen_list.append(home_feature_screen_array)
            home_feature_minimap_list.append(home_feature_minimap_array)
            home_feature_player_list.append(home_feature_player_array)
            home_feature_units_list.append(home_feature_units_array)
            home_available_actions_list.append([home_available_actions])
            home_memory_state_list.append(memory_state)
            home_carry_state_list.append(carry_state)
            game_loop_list.append(game_loop_array)
            last_action_type_list.append(np.array([last_action_type]))
            home_feature_screen_history_list.append(home_feature_screen_history_array)

            home_prediction = home_agent.act(home_feature_screen_array, home_feature_minimap_array,
                                                    home_feature_player_array, home_feature_units_array, 
                                                    home_available_actions_array, 
                                                    memory_state, carry_state,
                                                    game_loop_array, last_action_type_array)
            home_fn_pi = home_prediction['fn_out']
            home_arg_pis = home_prediction['args_out']
            home_next_memory_state = home_prediction['final_memory_state']
            home_next_carry_state = home_prediction['final_carry_state']

            home_fn_samples, home_arg_samples = sample_actions(home_available_actions, home_fn_pi, home_arg_pis)
            home_fn_id, home_arg_ids = mask_unused_argument_samples(home_fn_samples, home_arg_samples)
            home_fn_id_list.append(home_fn_id[0])

            home_arg_id_list = []
            for arg_type in home_arg_ids.keys():
                arg_id = home_arg_ids[arg_type]
                home_arg_id_list.append(arg_id)
            
            home_arg_ids_list.append(np.array([home_arg_id_list]))
            home_actions_list = actions_to_pysc2(home_fn_id, home_arg_ids, (32, 32))
            last_action_type = home_fn_id

            actions_list = [home_actions_list]
            next_state = env.step(actions_list)

            home_next_state = next_state[0]
            home_feature_next_screen = home_next_state[3]['feature_screen']
            home_feature_next_screen = utils.preprocess_screen(home_feature_next_screen)
            home_feature_next_screen = np.transpose(home_feature_next_screen, (1, 2, 0))
            home_feature_next_screen_array = np.array([home_feature_next_screen])
            home_feature_screens_t_1 = np.append(home_feature_next_screen_array, home_feature_screens_t[:3,:, :, :], axis=0)

            home_done = home_next_state[0]
            if home_done == StepType.LAST:
                home_done = True
            else:
                home_done = False

            state = next_state
            memory_state = home_next_memory_state
            carry_state =  home_next_carry_state
            home_feature_screens_t = home_feature_screens_t_1

            home_reward = float(home_next_state[1])
            home_rewards.append(home_reward)
            home_dones.append(home_done)

            home_score += home_reward
            if len(home_feature_screen_list) == 16:
                if arguments.training == True:
                  #print("len(last_action_type_list): ", len(last_action_type_list))
                  home_agent.reinforcement_replay(home_feature_screen_list, home_feature_minimap_list,
                                                          home_feature_player_list, home_feature_units_list, 
                                                          home_available_actions_list, home_fn_id_list, home_arg_ids_list, 
                                                          home_rewards, home_dones, 
                                                          home_memory_state_list, home_carry_state_list,
                                                          game_loop_list, last_action_type_list)

                home_feature_screen_list, home_feature_minimap_list = [], []
                home_feature_player_list, home_feature_units_list = [], []
                home_available_actions_list, last_action_type_list = [], []
                home_fn_id_list, home_arg_ids_list, home_rewards, home_dones = [], [], [], []
                home_memory_state_list, home_carry_state_list = [], []
                game_loop_list, home_feature_screen_history_list = [], []

        score_list.append(home_score)
        average = sum(score_list) / len(score_list)
        if average >= max_average:
          SAVING = "SAVING"
          max_average = average
          home_agent.save(workspace_path + '/Models/reinforcment_model')
        else:
          SAVING = ""

        PlotModel(home_score, episode)
        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(episode, EPISODES, home_score, average, SAVING))
        if episode < EPISODES:
            episode += 1

    env.close()   


def main():
  if arguments.sl_training:
    supervised_train(10000000)
    reinforcement_train(20000)
  else:
    reinforcement_train(10000)


if __name__ == "__main__":
  main()