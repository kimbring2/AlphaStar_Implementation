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
import argparse

import network
import utils

FLAGS = flags.FLAGS
FLAGS(['run_reinforcement_learning.py'])

parser = argparse.ArgumentParser(description='AlphaStar implementation')
parser.add_argument('--environment', type=str, default='MoveToBeacon', help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, help='root directory for checkpoint storage')
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=42, help='seed number')
parser.add_argument('--num_worker', type=int, default=5, help='worker number of A3C')
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


seed = arguments.seed
tf.random.set_seed(seed)
np.random.seed(seed)

workspace_path = arguments.workspace_path
writer = tf.summary.create_file_writer(workspace_path + "/tensorboard/4")

_NUM_FUNCTIONS = len(actions.FUNCTIONS)

is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def preprocess_available_actions(available_action):
    available_actions = np.zeros(_NUM_FUNCTIONS, dtype=np.float32)
    available_actions[available_action] = 1

    return available_actions


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
    args_out[arg_type] = int(arg_ids[arg_type][0])

  a_0 = fn_id[0]
  unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
  for arg_type in unused_types:
    args_out[arg_type] = -1

  return fn_id, args_out


def mask_unavailable_actions(available_actions, fn_pi):
  available_actions = tf.cast(available_actions, 'float32')
  fn_pi *= available_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)

  return fn_pi


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


mse_loss = tf.keras.losses.MeanSquaredError()

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

class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
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

        self.EPISODES, self.episode, self.max_average = 2000000, 0, 20.0
        self.lock = Lock()

        # Instantiate games and plot memory
        self.state_list, self.action_list, self.reward_list = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C'.format(self.env_name)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.ActorCritic = network.make_model(arguments.model_name)
        if arguments.load_model != None:
        	print("load_model")
        	model.load_weights(workspace_path + 'Models/' + arguments.load_model)

        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001,
                                                                            decay_steps=10000, decay_rate=0.94)
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate, epsilon=1e-5)

    @tf.function
    def act(self, feature_screen, feature_minimap, player, feature_units, memory_state, carry_state, game_loop, 
    		available_actions, build_queue, single_select, multi_select, score_cumulative):
        # Use the network to predict the next action to take, using the model
        input_dict = {'feature_screen': feature_screen, 'feature_minimap': feature_minimap,
                   	  'player': player, 'feature_units': feature_units, 
                      'memory_state': memory_state, 'carry_state': carry_state, 
                      'game_loop': game_loop, 'available_actions': available_actions, 
                      'build_queue': build_queue, 'single_select': single_select, 
                      'multi_select': multi_select, 'score_cumulative': score_cumulative}
        prediction = self.ActorCritic(input_dict, training=False)
        fn_pi = prediction['fn_out']
        arg_pis = prediction['args_out']
        memory_state = prediction['final_memory_state']
        carry_state = prediction['final_carry_state']

        fn_samples, arg_samples = sample_actions(available_actions, fn_pi, arg_pis)

        return fn_samples, arg_samples, memory_state, carry_state

    def compute_log_probs(self, probs, labels):
      labels = tf.maximum(labels, 0)
      labels = tf.cast(labels, 'int32')
      indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
      result = tf.gather_nd(probs, indices)
      result = tf.where(tf.equal(result, 0), tf.zeros_like(result), tf.math.log(tf.maximum(1e-12, result)))

      return result

    def mask_unavailable_actions(self, available_actions, fn_pi):
      available_actions = tf.cast(available_actions, 'float32')
      fn_pi *= available_actions
      fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)

      return fn_pi

    def safe_log(self, x):
      return tf.where(tf.equal(x, 0), tf.zeros_like(x), tf.math.log(tf.maximum(1e-12, x)))

    def discount_rewards(self, reward, dones):
      # Compute the gamma-discounted rewards over an episode
      gamma = 0.99    # discount rate
      running_add = 0
      discounted_r = np.zeros_like(reward)
      for i in reversed(range(0, len(reward))):
          running_add = running_add * gamma * (1 - dones[i]) + reward[i]
          discounted_r[i] = running_add

      if np.std(discounted_r) != 0:
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation

      return discounted_r

    @tf.function
    def get_loss(self, feature_screen_array, feature_minimap_array, player_array, feature_units_array,
          	     available_actions_array, game_loop_array, build_queue_array, single_select_array,
          	     multi_select_array, score_cumulative_array, memory_state, carry_state, discounted_r_array, 
          	     fn_id_array, arg_ids_array):
      batch_size = feature_screen_array.shape[0]

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

      values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
      for i in range(0, batch_size):
        input_dict = {'feature_screen': tf.expand_dims(feature_screen_array[i,:,:,:], 0), 
                      'feature_minimap': tf.expand_dims(feature_minimap_array[i,:,:,:], 0),
                      'player': tf.expand_dims(player_array[i,:], 0), 
                      'feature_units': tf.expand_dims(feature_units_array[i,:,:], 0), 
                      'memory_state': memory_state, 'carry_state': carry_state, 
                      'game_loop': tf.expand_dims(game_loop_array[i,:], 0), 
                      'available_actions': tf.expand_dims(available_actions_array[i,:], 0), 
                      'build_queue': tf.expand_dims(build_queue_array[i], 0), 
                      'single_select': tf.expand_dims(single_select_array[i], 0), 
                      'multi_select': tf.expand_dims(multi_select_array[i], 0), 
                      'score_cumulative': tf.expand_dims(score_cumulative_array[i], 0)}

        prediction = self.ActorCritic(input_dict, training=True)
        fn_pi = prediction['fn_out']
        args_pi = prediction['args_out']
        value_estimate = prediction['value']
        memory_state = prediction['final_memory_state']
        carry_state = prediction['final_carry_state']

        fn_probs = fn_probs.write(i, fn_pi[0])
        values = values.write(i, tf.squeeze(value_estimate))

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

      values = values.stack()

      discounted_r_array = tf.cast(discounted_r_array, 'float32')
      advantage = discounted_r_array - values

      fn_probs = self.mask_unavailable_actions(available_actions_array, fn_probs) # TODO: this should be unneccessary

      fn_log_prob = self.compute_log_probs(fn_probs, fn_id_array)
      log_prob = fn_log_prob
      for index, arg_type in enumerate(actions.TYPES):
        arg_id = arg_ids_array[:,index]

        if arg_type.name == 'screen':
          arg_log_prob = self.compute_log_probs(screen_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
        elif arg_type.name == 'minimap':
          arg_log_prob = self.compute_log_probs(minimap_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
        elif arg_type.name == 'screen2':
          arg_log_prob = self.compute_log_probs(screen2_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
        elif arg_type.name == 'queued':
          arg_log_prob = self.compute_log_probs(queued_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
        elif arg_type.name == 'control_group_act':
          arg_log_prob = self.compute_log_probs(control_group_act_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	
        elif arg_type.name == 'control_group_id':
          arg_log_prob = self.compute_log_probs(control_group_id_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	
        elif arg_type.name == 'select_point_act':
          arg_log_prob = self.compute_log_probs(select_point_act_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	 
        elif arg_type.name == 'select_add':
          arg_log_prob = self.compute_log_probs(select_add_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	  
        elif arg_type.name == 'select_unit_act':
          arg_log_prob = self.compute_log_probs(select_unit_act_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	
        elif arg_type.name == 'select_unit_id':
          arg_log_prob = self.compute_log_probs(select_unit_id_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	 
        elif arg_type.name == 'select_worker':
          arg_log_prob = self.compute_log_probs(select_worker_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	
        elif arg_type.name == 'build_queue_id':
          arg_log_prob = self.compute_log_probs(build_queue_id_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')	 
        elif arg_type.name == 'unload_id':
          arg_log_prob = self.compute_log_probs(unload_id_arg_probs, arg_id)
          arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')

        log_prob += arg_log_prob

      actor_loss = -tf.math.reduce_mean(log_prob * tf.stop_gradient(advantage)) 
      actor_loss = tf.cast(actor_loss, 'float32')

      critic_loss = mse_loss(values, discounted_r_array)
      critic_loss = tf.cast(critic_loss, 'float32')
        
      total_loss = actor_loss + critic_loss * 0.5

      return total_loss

    def replay(self, feature_screen_list, feature_minimap_list, player_list, feature_units_list, available_actions_list, 
    		   game_loop_list, build_queue_list, single_select_list, multi_select_list, score_cumulative_list,
    		   memory_state, carry_state, fn_id_list, arg_ids_list, rewards, dones):
        feature_screen_array = np.vstack(feature_screen_list)
        feature_minimap_array = np.vstack(feature_minimap_list)
        player_array = np.vstack(player_list)
        feature_units_array = np.vstack(feature_units_list)
        available_actions_array = np.vstack(available_actions_list)
        game_loop_array = np.vstack(game_loop_list)
        build_queue_array = np.vstack(build_queue_list)
        single_select_array = np.vstack(single_select_list)
        multi_select_array = np.vstack(multi_select_list)
        score_cumulative_array = np.vstack(score_cumulative_list)

        fn_id_array = np.array(fn_id_list)
        arg_ids_array = np.array(arg_ids_list)

        # Compute discounted rewards
        discounted_r_array = self.discount_rewards(rewards, dones)
        with tf.GradientTape() as tape:
          total_loss = self.get_loss(feature_screen_array, feature_minimap_array, player_array, feature_units_array,
          							 available_actions_array, game_loop_array, build_queue_array, single_select_array,
          							 multi_select_array, score_cumulative_array, memory_state, carry_state,
                                     discounted_r_array, fn_id_array, arg_ids_array)

        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        grads_norm = tf.linalg.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, arguments.gradient_clipping)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

        return total_loss, grads_norm

    def load(self):
        self.ActorCritic.load_weights(workspace_path + 'Models/' + arguments.load_model)

    def save(self):
        self.ActorCritic.save_weights(workspace_path + 'Models/' + arguments.load_model)

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
        max_average = 5.0
        total_step = 0
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.reset(env)

            feature_screen_list, feature_minimap_list, player_list, feature_units_list = [], [], [], []
            available_actions_list, game_loop_list, build_queue_list = [], [], []
            single_select_list, multi_select_list, score_cumulative_list = [], [], []
            fn_id_list, arg_ids_list, rewards, dones = [], [], [], []

            memory_state = np.zeros([1,256], dtype=np.float32)
            carry_state = np.zeros([1,256], dtype=np.float32)

            initial_memory_state = memory_state
            initial_carry_state = carry_state
            while not done:
                feature_screen = state[0][3]['feature_screen']
                feature_screen = utils.preprocess_screen(feature_screen)
                feature_screen = np.transpose(feature_screen, (1, 2, 0))

                feature_minimap = state[0][3]['feature_minimap']
                feature_minimap = utils.preprocess_minimap(feature_minimap)
                feature_minimap = np.transpose(feature_minimap, (1, 2, 0))

                player = state[0][3]['player']
                player = utils.preprocess_player(player)

                feature_units = state[0][3]['feature_units']
                feature_units = utils.preprocess_feature_units(feature_units, feature_screen_size)

                available_actions = state[0][3]['available_actions']
                available_actions = utils.preprocess_available_actions(available_actions)

                game_loop = state[0][3]['game_loop']

                build_queue = state[0][3]['build_queue']
                build_queue = utils.preprocess_build_queue(build_queue)

                single_select = state[0][3]['single_select']
                single_select = utils.preprocess_single_select(single_select)

                multi_select = state[0][3]['multi_select']
                multi_select = utils.preprocess_multi_select(multi_select)

                score_cumulative = state[0][3]['score_cumulative']
                score_cumulative = utils.preprocess_score_cumulative(score_cumulative)

                feature_screen_array = np.array([feature_screen])
                feature_minimap_array = np.array([feature_minimap])
                player_array = np.array([player])
                feature_units_array = np.array([feature_units])
                available_actions_array = np.array([available_actions])
                game_loop_array = np.array([game_loop])
                build_queue_array = np.array([build_queue])
                single_select_array = np.array([single_select])
                multi_select_array = np.array([multi_select])
                score_cumulative_array = np.array([score_cumulative])

                feature_screen_list.append(feature_screen_array)
                feature_minimap_list.append(feature_minimap_array)
                player_list.append(player_array)
                feature_units_list.append(feature_units_array)
                available_actions_list.append([available_actions])
                game_loop_list.append(game_loop_array)
                build_queue_list.append(build_queue_array)
                single_select_list.append(single_select_array)
                multi_select_list.append(multi_select_array)
                score_cumulative_list.append(score_cumulative_array)

                prediction = agent.act(feature_screen_array, feature_minimap_array, player_array, feature_units_array, 
                					   memory_state, carry_state, game_loop_array, available_actions_array, build_queue_array,
                					   single_select_array, multi_select_array, score_cumulative_array)
                fn_samples = prediction[0]
                arg_samples = prediction[1]
                memory_state = prediction[2]
                carry_state = prediction[3]
                fn_id, arg_ids = mask_unused_argument_samples(fn_samples, arg_samples)
                fn_id_list.append(fn_id.numpy()[0])

                arg_id_list = []
                for arg_type in arg_ids.keys():
                  arg_id = arg_ids[arg_type]
                  arg_id_list.append(arg_id)

                arg_ids_list.append(np.array(arg_id_list))
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
                    total_step += 1
                    self.lock.acquire()
                    total_loss, grads_norm = self.replay(feature_screen_list, feature_minimap_list, player_list, feature_units_list, 
                                                         available_actions_list, game_loop_list, build_queue_list, single_select_list, 
                                                         multi_select_list, score_cumulative_list, initial_memory_state, 
                                                         initial_carry_state, fn_id_list, arg_ids_list, rewards, dones)
                    self.lock.release()

                    initial_memory_state = memory_state
                    initial_carry_state = carry_state

                    feature_screen_list, feature_minimap_list, player_list, feature_units_list = [], [], [], []
                    available_actions_list, game_loop_list, build_queue_list = [], [], []
                    single_select_list, multi_select_list, score_cumulative_list = [], [], []
                    fn_id_list, arg_ids_list, rewards, dones = [], [], [], []

            score_list.append(score)
            average = sum(score_list) / len(score_list)
            
            if thread == 0:
              with writer.as_default():
                # other model code would go here
                tf.summary.scalar("grads_norm", grads_norm, step=total_step)
                tf.summary.scalar("total_loss", total_loss, step=total_step)
                tf.summary.scalar("average", average, step=total_step)
                writer.flush()

            # Update episode count
            with self.lock:
                #self.PlotModel(score, self.episode)
                if average >= max_average:
                  max_average = average
                  if thread == 0:
                    self.save()
                    SAVING = "SAVING"
                else:
                  SAVING = ""

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
    agent.train(n_threads=arguments.num_worker) # use as A3C