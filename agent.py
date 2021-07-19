from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape
from tensorflow_probability.python.distributions import kullback_leibler

from absl import flags

import utils

mse_loss = tf.keras.losses.MeanSquaredError()
kl_loss = tf.keras.losses.KLDivergence()
cce = tf.keras.losses.CategoricalCrossentropy()
'''
FLAGS = flags.FLAGS
FLAGS(['agent.py'])

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
arguments = parser.parse_args()
'''

class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, network, learning_rate, gradient_clipping):
        # Instantiate games and plot memory
        '''
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
              visualize=arguments.visualize)
        '''
        # Create Actor-Critic network model
        self.ActorCritic = network
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.94,
            staircase=True)

        self.optimizer_rl = tf.keras.optimizers.RMSprop(lr_schedule, epsilon=1e-5)
        self.optimizer_sl = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-5)

        #if arguments.load == True:
        #  self.load()

    def run(self):
        state = self.env.reset()
        while not done:
            home_state = state[0]

            home_feature_screen = home_state[3]['feature_screen']
            home_feature_screen = utils.preprocess_screen(home_feature_screen)
            home_feature_screen = np.transpose(home_feature_screen, (1, 2, 0))

            home_feature_player = home_state[3]['player']
            home_feature_player = utils.preprocess_player(home_feature_player)

            home_available_actions = home_state[3]['available_actions']
            home_available_actions = utils.preprocess_available_actions(home_available_actions)

            home_feature_units = home_state[3]['feature_units']
            home_feature_units = utils.preprocess_feature_units(home_feature_units, feature_screen_size)
        
    @tf.function
    def act(self, feature_screen_array, feature_player_array, feature_units_array, available_actions_array, memory_state, carry_state, 
             game_loop_array, last_action_type_array):
        # Use the network to predict the next action to take, using the model
        prediction = self.ActorCritic(feature_screen_array, feature_player_array, feature_units_array, memory_state, carry_state, 
                                          game_loop_array, available_actions_array, last_action_type_array, training=False)
        return prediction
    
    @tf.function
    def discount_rewards(self, reward, dones):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        reward_copy = np.array(reward)
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma * (1 - dones[i]) + reward[i]
            discounted_r[i] = running_add

        return discounted_r
    
    @tf.function
    def compute_entropy(self, probs):
      return -tf.reduce_sum(self.safe_log(probs) * probs, axis=-1)

    @tf.function
    def compute_log_probs(self, probs, labels):
      labels = tf.maximum(labels, 0)
      labels = tf.cast(labels, 'int32')
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
    def safe_div(self, numerator, denominator, name="value"):
      return tf.where(
        tf.greater(denominator, 0),
        tf.math.divide(numerator, tf.where(
            tf.math.equal(denominator, 0),
            tf.ones_like(denominator), denominator)),
        tf.zeros_like(numerator),
        name=name)

    @tf.function
    def safe_log(self, x):
      return tf.where(
          tf.equal(x, 0),
          tf.zeros_like(x),
          tf.math.log(tf.maximum(1e-12, x)))
    
    def supervised_replay(self, replay_feature_screen_list, replay_feature_player_list, replay_feature_units_list, 
                             replay_available_actions_list, replay_fn_id_list, replay_args_ids_list,
                             memory_state_list, carry_state_list,
                             replay_game_loop_list, delay_list, last_action_type_list):
        replay_feature_screen_array = tf.concat(replay_feature_screen_list, 0)
        replay_feature_player_array = tf.concat(replay_feature_player_list, 0)
        replay_feature_units_array = tf.concat(replay_feature_units_list, 0)
        replay_game_loop_array = tf.concat(replay_game_loop_list, 0)
        delay_array = tf.concat(delay_list, 0)
        last_action_type_array = tf.concat(last_action_type_list, 0)
        replay_available_actions_array = tf.concat(replay_available_actions_list, 0)
        replay_fn_id_array = tf.concat(replay_fn_id_list, 0)
        replay_arg_ids_array = tf.concat(replay_args_ids_list, 0)

        home_memory_state_array = tf.concat(memory_state_list, 0)
        home_carry_state_array = tf.concat(carry_state_list, 0)
        with tf.GradientTape() as tape:
          prediction = self.ActorCritic(replay_feature_screen_array, replay_feature_player_array, replay_feature_units_array,
                                            home_memory_state_array, home_carry_state_array, 
                                            replay_game_loop_array, replay_available_actions_array, last_action_type_array, 
                                            training=True)
          fn_pi = prediction[0]
          arg_pis = prediction[1]
          delay = prediction[5]

          batch_size = fn_pi.shape[0]

          replay_fn_id_array_onehot = tf.one_hot(replay_fn_id_array, 573)
          replay_fn_id_array_onehot = tf.reshape(replay_fn_id_array_onehot, (batch_size, 573))

          #print("replay_fn_id_array_onehot: ", replay_fn_id_array_onehot)
          print("replay_fn_id_array: ", replay_fn_id_array)
          print("tf.argmax(fn_pi, 1): ", tf.argmax(fn_pi, 1))
          fn_id_loss = cce(replay_fn_id_array_onehot, fn_pi)
          arg_ids_loss = 0 
          for index, arg_type in enumerate(actions.TYPES):
            replay_arg_id = replay_arg_ids_array[:,index]
            arg_pi = arg_pis[arg_type]
            replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, arg_pi.shape[1])
            arg_id_loss = cce(replay_arg_id_array_onehot, arg_pi)

            arg_ids_loss += arg_id_loss

          delay_array_onehot = tf.one_hot(delay_array, 500)
          delay_array_onehot = tf.reshape(delay_array_onehot, (batch_size, 500))

          print("delay_array: ", delay_array)
          print("tf.argmax(delay, 1): ", tf.argmax(delay, 1))
          delay_loss = cce(delay, delay_array_onehot)
          print("fn_id_loss: ", fn_id_loss)
          print("arg_ids_loss: ", arg_ids_loss)
          print("delay_loss: ", delay_loss)
          regularization_loss = tf.reduce_sum(self.ActorCritic.losses)

          total_loss = fn_id_loss + arg_ids_loss + 1e-5 * regularization_loss + 0.1 * delay_loss

        print("total_loss: ", total_loss)
        print("")
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.gradient_clipping)
        self.optimizer_sl.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

    def reinforcement_replay(self, feature_screen_list, feature_player_list, feature_units_list, available_actions_list, 
                                fn_id_list, arg_ids_list, 
                                rewards, dones, 
                                home_memory_state_list, home_carry_state_list, 
                                game_loop_list, last_action_type_list):
        feature_screen_array = tf.concat(feature_screen_list, 0)
        feature_player_array = tf.concat(feature_player_list, 0)
        feature_units_array = tf.concat(feature_units_list, 0)
        game_loop_array = tf.concat(game_loop_list, 0)
        #delay_array = tf.concat(delay_list, 0)
        last_action_type_array = tf.concat(last_action_type_list, 0)
        available_actions_array = tf.concat(available_actions_list, 0)
        arg_ids_array = tf.concat(arg_ids_list, 0)

        home_memory_state_array = tf.concat(home_memory_state_list, 0)
        home_carry_state_array = tf.concat(home_carry_state_list, 0)

        # Compute discounted rewards
        discounted_r_array = self.discount_rewards(rewards, dones)
        with tf.GradientTape() as tape:
          prediction = self.ActorCritic(feature_screen_array, feature_player_array, feature_units_array, 
                                            home_memory_state_array, home_carry_state_array, 
                                            game_loop_array, available_actions_array, last_action_type_array,
                                            training=True)
          fn_pi = prediction[0]
          arg_pis = prediction[1]
          value_estimate = prediction[2]
          #delay_pi = prediction[5]

          discounted_r_array = tf.cast(discounted_r_array, 'float32')
          advantage = discounted_r_array - tf.stack(value_estimate)[:, 0]

          #print("available_actions_array: ", available_actions_array)
          #print("fn_pi: ", fn_pi)
          fn_pi = self.mask_unavailable_actions(available_actions_array, fn_pi) # TODO: this should be unneccessary
          fn_log_prob = self.compute_log_probs(fn_pi, fn_id_list)
          log_prob = fn_log_prob
          for index, arg_type in enumerate(actions.TYPES):
            arg_id = arg_ids_array[:,index]
            arg_pi = arg_pis[arg_type]
            arg_log_prob = self.compute_log_probs(arg_pi, arg_id)
            arg_log_prob *= tf.cast(tf.not_equal(arg_id, -1), 'float32')
            log_prob += arg_log_prob

          #print("delay_pi: ", delay_pi)
          #print("delay_list: ", delay_list)
          #delay_log_prob = self.compute_log_probs(delay_pi, delay_list)
          #delay_log_prob *= tf.cast(tf.not_equal(delay_list, -1), 'float32')
          #log_prob += delay_log_prob

          actor_loss = -tf.math.reduce_mean(log_prob * advantage) 
          actor_loss = tf.cast(actor_loss, 'float32')

          critic_loss = mse_loss(tf.stack(value_estimate)[:, 0] , discounted_r_array)
          critic_loss = tf.cast(critic_loss, 'float32')
        
          entropy_loss = tf.reduce_mean(self.compute_entropy(fn_pi))
          for index, arg_type in enumerate(actions.TYPES):
            arg_id = arg_ids_array[:,index]
            arg_pi = arg_pis[arg_type]
            batch_mask = tf.cast(tf.not_equal(arg_id, -1), 'float32')
            arg_entropy = self.safe_div(
               tf.reduce_sum(self.compute_entropy(arg_pi) * batch_mask),
               tf.reduce_sum(batch_mask))
           
            entropy_loss += arg_entropy
          
          total_loss = actor_loss + 0.5 * critic_loss - 1e-3 * entropy_loss

        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.gradient_clipping)
        self.optimizer_rl.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

    def load(self, path):
        self.ActorCritic.load_weights(path)
    
    def save(self, path):
        self.ActorCritic.save_weights(path)