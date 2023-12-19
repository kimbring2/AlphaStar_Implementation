import time
import math
import zmq
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTMCell
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import threading
import random
import collections
import argparse
from absl import flags
from absl import logging
from typing import Any, List, Sequence, Tuple

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import gym

import network
from parametric_distribution import get_parametric_distribution_for_action_space

parser = argparse.ArgumentParser(description='PySC2 IMPALA Server')
parser.add_argument('--env_num', type=int, default=2, help='ID of environment')
parser.add_argument('--model_name', type=str, default='fullyconv', 
                    choices=["fullyconv", "alphastar", "relationalfullyconv"], help='model name')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--gradient_clipping', type=float, default=10.0, help='gradient clipping value')
arguments = parser.parse_args()

tfd = tfp.distributions

writer = tf.summary.create_file_writer("tensorboard_learner")

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("gpus: ", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e) 
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


socket_list = []
for i in range(0, arguments.env_num):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(5555 + i))

    socket_list.append(socket)


_NUM_FUNCTIONS = len(actions.FUNCTIONS)

batch_size = 1
unroll_length = 60

screen_width = 16
minimap_width = 16

## state size
feature_screen_size = (screen_width,screen_width,24)
feature_minimap_size = (minimap_width,minimap_width,7)
player_size = 11
feature_units_size = (50,8)
available_actions_size = (573)
game_loop_size = 1
build_queue_size = 5
single_select_size = 3
multi_select_size = 10
score_cumulative_size = 13
act_history_size = (16, _NUM_FUNCTIONS)

## action size
fn_action_size = _NUM_FUNCTIONS
screen_action_size = screen_width*screen_width
minimap_action_size = minimap_width*minimap_width
screen2_action_size = screen_width*screen_width
queued_action_size = 2
control_group_act_action_size = 5
control_group_id_action_size = 10
select_point_act_action_size = 4
select_add_action_size = 2
select_unit_act_action_size = 4
select_unit_id_action_size = 500
select_worker_action_size = 4
build_queue_id_action_size = 10
unload_id_action_size = 500

## etc size
memory_state = (256)
carry_state = (256)

queue = tf.queue.FIFOQueue(1, dtypes=[tf.int32, tf.float32, tf.bool,

                                      tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,

                                      tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, 

                                      tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 

                                      tf.float32, tf.float32], 
                           shapes=[[unroll_length+1], [unroll_length+1], [unroll_length+1],

                                   [unroll_length+1,*feature_screen_size], [unroll_length+1,*feature_minimap_size], [unroll_length+1,player_size], [unroll_length+1,*feature_units_size], [unroll_length+1,available_actions_size],
                                   [unroll_length+1,game_loop_size], [unroll_length+1,build_queue_size], [unroll_length+1,single_select_size], [unroll_length+1,multi_select_size], [unroll_length+1,score_cumulative_size],
                                   [unroll_length+1,*act_history_size],

                                   [unroll_length+1,fn_action_size], [unroll_length+1,screen_action_size], [unroll_length+1,minimap_action_size], [unroll_length+1,screen2_action_size], [unroll_length+1,queued_action_size], 
                                   [unroll_length+1,control_group_act_action_size], [unroll_length+1,control_group_id_action_size], [unroll_length+1,select_point_act_action_size], [unroll_length+1,select_add_action_size], 
                                   [unroll_length+1,select_unit_act_action_size], [unroll_length+1,select_unit_id_action_size], [unroll_length+1,select_worker_action_size], [unroll_length+1,build_queue_id_action_size], 
                                   [unroll_length+1,unload_id_action_size], 

                                   [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1], 
                                   [unroll_length+1], [unroll_length+1], [unroll_length+1], [unroll_length+1],

                                   [unroll_length+1,256], [unroll_length+1,256]])

#Unroll = collections.namedtuple('Unroll', 'env_id reward done observation policy action memory_state carry_state')
Unroll = collections.namedtuple('Unroll', 'env_id reward done \
   feature_screen feature_minimap player feature_units available_actions game_loop build_queue single_select multi_select score_cumulative act_history \
   fn_policy, screen_policy, minimap_policy, screen2_policy, queued_policy, control_group_act_policy, control_group_id_policy, select_point_act_policy, select_add_policy, select_unit_act_policy, select_unit_id_policy, select_worker_policy, build_queue_id_policy, unload_id_policy \
   fn, screen, minimap, screen2, queued, control_group_act, control_group_id, select_point_act, select_add, select_unit_act, select_unit_id, select_worker, build_queue_id, unload_id \
   memory_state carry_state')

model = network.make_model(arguments.model_name)

if arguments.pretrained_model != None:
    print("Load Pretrained Model")
    #model.load_weights("model/" + arguments.pretrained_model)
    
#model.set_weights(sl_model.get_weights())

num_action_repeats = 1
total_environment_frames = int(4e7)

iter_frame_ratio = (batch_size * unroll_length * num_action_repeats)
final_iteration = int(math.ceil(total_environment_frames / iter_frame_ratio))
    
lr = tf.keras.optimizers.schedules.PolynomialDecay(0.0001, final_iteration, 0)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.99, epsilon=1e-05)

fn_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(fn_action_size))
screen_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(screen_action_size))
minimap_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(minimap_action_size))
screen2_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(screen2_action_size))
queued_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(queued_action_size))
control_group_act_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(control_group_act_action_size))
control_group_id_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(control_group_id_action_size))
select_point_act_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(select_point_act_action_size))
select_add_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(select_add_action_size))
select_unit_act_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(select_unit_act_action_size))
select_unit_id_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(select_unit_id_action_size))
select_worker_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(select_worker_action_size))
build_queue_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(build_queue_id_action_size))
unload_id_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(unload_id_action_size))


def safe_log(x):
    return tf.where(tf.equal(x, 0), tf.zeros_like(x), tf.math.log(tf.maximum(1e-12, x)))


def safe_div(numerator, denominator, name="value"):
    return tf.where(tf.greater(denominator, 0), tf.math.divide(numerator, tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)), tf.zeros_like(numerator), name=name)


def compute_log_probs(logits, labels):
    probs = tf.nn.softmax(logits)
    labels = tf.maximum(labels, 0)
    labels = tf.cast(labels, 'int32')
    indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
    result = tf.gather_nd(probs, indices)
    result = tf.where(tf.equal(result, 0), tf.zeros_like(result), tf.math.log(tf.maximum(1e-12, result)))
    
    return result


def vtrace(states, learner_policies, learner_values, agent_policies, actions, rewards, dones, parametric_action_distribution, masking=True):
    learner_policies = tf.reshape(learner_policies, [states.shape[0], states.shape[1], -1])
    learner_values = tf.reshape(learner_values, [states.shape[0], states.shape[1], -1])
    
    actions = actions[:-1]
    rewards = rewards[1:]
    dones = dones[1:]
    
    learner_values = tf.squeeze(learner_values, axis=2)
        
    bootstrap_value = learner_values[-1]
    learner_values = learner_values[:-1]
        
    discounting = 0.99
    discounts = tf.cast(~dones, tf.float32) * discounting

    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        
    target_action_log_probs = parametric_action_distribution.log_prob(learner_policies[:-1], tf.maximum(actions, 0))
    behaviour_action_log_probs = parametric_action_distribution.log_prob(agent_policies[:-1], tf.maximum(actions, 0))

    batch_mask = tf.cast(tf.not_equal(actions, -1), 'float32')
    if masking:
        target_action_log_probs *= batch_mask
        behaviour_action_log_probs *= batch_mask

    lambda_ = 1.0
        
    log_rhos = target_action_log_probs - behaviour_action_log_probs
        
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(learner_values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
        
    clip_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
    clip_pg_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
        
    rhos = tf.math.exp(log_rhos)
        
    clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
        
    cs = tf.minimum(1.0, rhos, name='cs')
    cs *= tf.convert_to_tensor(lambda_, dtype=tf.float32)

    values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    
    acc = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = []
    for i in range(int(discounts.shape[0]) - 1, -1, -1):
        discount, c, delta = discounts[i], cs[i], deltas[i]
        acc = delta + discount * c * acc
        vs_minus_v_xs.append(acc)  
        
    vs_minus_v_xs = vs_minus_v_xs[::-1]
        
    vs = tf.add(vs_minus_v_xs, values, name='vs')
    vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
        
    pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
        
    vs = tf.stop_gradient(vs)
    pg_advantages = tf.stop_gradient(pg_advantages)

    actor_loss = -tf.reduce_mean(target_action_log_probs * pg_advantages)

    baseline_cost = 0.1
    v_error = values - vs
    critic_loss = tf.square(v_error)
    critic_loss = baseline_cost * 0.5 * tf.reduce_mean(critic_loss)
    
    entropy_loss = parametric_action_distribution.entropy(learner_policies[:-1])

    if masking:
        entropy_loss *= batch_mask

    entropy_loss = tf.reduce_mean(entropy_loss)
    entropy_loss = safe_div(entropy_loss, tf.reduce_sum(batch_mask))
    entropy_loss = 0.0025 * -entropy_loss

    total_loss = actor_loss + critic_loss + entropy_loss

    return total_loss


def update(feature_screen_states, feature_minimap_states, player_states, feature_units_states, available_actions_states, game_loop_states, build_queue_states, 
           single_select_states, multi_select_states, score_cumulative_states, act_history_states,

           fn_actions, screen_actions, minimap_actions, screen2_actions, queued_actions, control_group_act_actions, control_group_id_actions, select_point_act_actions, 
           select_add_actions, select_unit_act_actions, select_unit_id_actions, select_worker_actions, build_queue_id_actions, unload_id_actions,

           fn_policies, screen_policies, minimap_policies, screen2_policies, queued_policies, control_group_act_policies, control_group_id_policies, select_point_act_policies, 
           select_add_policies, select_unit_act_policies, select_unit_id_policies, select_worker_policies, build_queue_id_policies, unload_id_policies,
           
           rewards, dones, memory_states, carry_states):
    feature_screen_states = tf.transpose(feature_screen_states, perm=[1, 0, 2, 3, 4])
    feature_minimap_states = tf.transpose(feature_minimap_states, perm=[1, 0, 2, 3, 4])
    player_states = tf.transpose(player_states, perm=[1, 0, 2])
    feature_units_states = tf.transpose(feature_units_states, perm=[1, 0, 2, 3])
    available_actions_states = tf.transpose(available_actions_states, perm=[1, 0, 2])
    game_loop_states = tf.transpose(game_loop_states, perm=[1, 0, 2])
    build_queue_states = tf.transpose(build_queue_states, perm=[1, 0, 2])
    single_select_states = tf.transpose(single_select_states, perm=[1, 0, 2])
    multi_select_states = tf.transpose(multi_select_states, perm=[1, 0, 2])
    score_cumulative_states = tf.transpose(score_cumulative_states, perm=[1, 0, 2])
    act_history_states = tf.transpose(act_history_states, perm=[1, 0, 2, 3])

    fn_actions = tf.transpose(fn_actions, perm=[1, 0])
    screen_actions = tf.transpose(screen_actions, perm=[1, 0])
    minimap_actions = tf.transpose(minimap_actions, perm=[1, 0])
    screen2_actions = tf.transpose(screen2_actions, perm=[1, 0])
    queued_actions = tf.transpose(queued_actions, perm=[1, 0])
    control_group_act_actions = tf.transpose(control_group_act_actions, perm=[1, 0])
    control_group_id_actions = tf.transpose(control_group_id_actions, perm=[1, 0])
    select_point_act_actions = tf.transpose(select_point_act_actions, perm=[1, 0])
    select_add_actions = tf.transpose(select_add_actions, perm=[1, 0])
    select_unit_act_actions = tf.transpose(select_unit_act_actions, perm=[1, 0])
    select_unit_id_actions = tf.transpose(select_unit_id_actions, perm=[1, 0])
    select_worker_actions = tf.transpose(select_worker_actions, perm=[1, 0])
    build_queue_id_actions = tf.transpose(build_queue_id_actions, perm=[1, 0])
    unload_id_actions = tf.transpose(unload_id_actions, perm=[1, 0])

    fn_policies = tf.transpose(fn_policies, perm=[1, 0, 2])
    screen_policies = tf.transpose(screen_policies, perm=[1, 0, 2])
    minimap_policies = tf.transpose(minimap_policies, perm=[1, 0, 2])
    screen2_policies = tf.transpose(screen2_policies, perm=[1, 0, 2])
    queued_policies = tf.transpose(queued_policies, perm=[1, 0, 2])
    control_group_act_policies = tf.transpose(control_group_act_policies, perm=[1, 0, 2])
    control_group_id_policies = tf.transpose(control_group_id_policies, perm=[1, 0, 2])
    select_point_act_policies = tf.transpose(select_point_act_policies, perm=[1, 0, 2])
    select_add_policies = tf.transpose(select_add_policies, perm=[1, 0, 2])
    select_unit_act_policies = tf.transpose(select_unit_act_policies, perm=[1, 0, 2])
    select_unit_id_policies = tf.transpose(select_unit_id_policies, perm=[1, 0, 2])
    select_worker_policies = tf.transpose(select_worker_policies, perm=[1, 0, 2])
    build_queue_id_policies = tf.transpose(build_queue_id_policies, perm=[1, 0, 2])
    unload_id_policies = tf.transpose(unload_id_policies, perm=[1, 0, 2])

    rewards = tf.transpose(rewards, perm=[1, 0])
    dones = tf.transpose(dones, perm=[1, 0])
    memory_states = tf.transpose(memory_states, perm=[1, 0, 2])
    carry_states = tf.transpose(carry_states, perm=[1, 0, 2])
    
    batch_size = feature_screen_states.shape[0]
        
    online_variables = model.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(online_variables)
               
        learner_fn_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_screen_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_minimap_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_screen2_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_queued_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_control_group_act_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_control_group_id_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_select_point_act_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_select_add_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_select_unit_act_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_select_unit_id_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_select_worker_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_build_queue_id_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_unload_id_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        learner_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        memory_state = memory_states[0]
        carry_state = carry_states[0]
        for i in tf.range(0, batch_size):
            prediction = model([feature_screen_states[i,:,:,:], feature_minimap_states[i,:,:,:], player_states[i,:], feature_units_states[i,:,:], game_loop_states[i,:],
                                available_actions_states[i,:], build_queue_states[i], single_select_states[i], multi_select_states[i], score_cumulative_states[i],
                                act_history_states[i,:,:], memory_state, carry_state], training=True)

            learner_fn_policies = learner_fn_policies.write(i, prediction[0])
            learner_screen_policies = learner_screen_policies.write(i, prediction[1])
            learner_minimap_policies = learner_minimap_policies.write(i, prediction[2])
            learner_screen2_policies = learner_screen2_policies.write(i, prediction[3])
            learner_queued_policies = learner_queued_policies.write(i, prediction[4])
            learner_control_group_act_policies = learner_control_group_act_policies.write(i, prediction[5])
            learner_control_group_id_policies = learner_control_group_id_policies.write(i, prediction[6])
            learner_select_point_act_policies = learner_select_point_act_policies.write(i, prediction[7])
            learner_select_add_policies = learner_select_add_policies.write(i, prediction[8])
            learner_select_unit_act_policies = learner_select_unit_act_policies.write(i, prediction[9])
            learner_select_unit_id_policies = learner_select_unit_id_policies.write(i, prediction[10])
            learner_select_worker_policies = learner_select_worker_policies.write(i, prediction[11])
            learner_build_queue_id_policies = learner_build_queue_id_policies.write(i, prediction[12])
            learner_unload_id_policies = learner_unload_id_policies.write(i, prediction[13])

            learner_values = learner_values.write(i, prediction[14])
            
            memory_state = prediction[15]
            carry_state = prediction[16]

        learner_fn_policies = learner_fn_policies.stack()
        learner_fn_policies = mask_unavailable_actions(available_actions_states, learner_fn_policies)

        learner_screen_policies = learner_screen_policies.stack()
        learner_minimap_policies = learner_minimap_policies.stack()
        learner_screen2_policies = learner_screen2_policies.stack()
        learner_queued_policies = learner_queued_policies.stack()
        learner_control_group_act_policies = learner_control_group_act_policies.stack()
        learner_control_group_id_policies = learner_control_group_id_policies.stack()
        learner_select_point_act_policies = learner_select_point_act_policies.stack()
        learner_select_add_policies = learner_select_add_policies.stack()
        learner_select_unit_act_policies = learner_select_unit_act_policies.stack()
        learner_select_unit_id_policies = learner_select_unit_id_policies.stack()
        learner_select_worker_policies = learner_select_worker_policies.stack()
        learner_build_queue_id_policies = learner_build_queue_id_policies.stack()
        learner_unload_id_policies = learner_unload_id_policies.stack()
        learner_values = learner_values.stack()

        fn_loss = vtrace(feature_screen_states, learner_fn_policies, learner_values, fn_policies, fn_actions, rewards, dones, fn_parametric_action_distribution, masking=False)
        screen_args_loss = vtrace(feature_screen_states, learner_screen_policies, learner_values, screen_policies, screen_actions, rewards, dones, screen_parametric_action_distribution)
        minimap_args_loss = vtrace(feature_screen_states, learner_minimap_policies, learner_values, minimap_policies, minimap_actions, rewards, dones, minimap_parametric_action_distribution)
        screen2_args_loss = vtrace(feature_screen_states, learner_screen2_policies, learner_values, screen2_policies, screen2_actions, rewards, dones, screen2_parametric_action_distribution)
        queued_args_loss = vtrace(feature_screen_states, learner_queued_policies, learner_values, queued_policies, queued_actions, rewards, dones, queued_parametric_action_distribution)
        control_group_act_args_loss = vtrace(feature_screen_states, learner_control_group_act_policies, learner_values, control_group_act_policies, control_group_act_actions, rewards, dones, control_group_act_parametric_action_distribution)
        control_group_id_args_loss = vtrace(feature_screen_states, learner_control_group_id_policies, learner_values, control_group_id_policies, control_group_id_actions, rewards, dones, control_group_id_parametric_action_distribution)
        select_point_act_args_loss = vtrace(feature_screen_states, learner_select_point_act_policies, learner_values, select_point_act_policies, select_point_act_actions, rewards, dones, select_point_act_parametric_action_distribution)
        select_add_args_loss = vtrace(feature_screen_states, learner_select_add_policies, learner_values, select_add_policies, select_add_actions, rewards, dones, select_add_parametric_action_distribution)
        select_unit_act_args_loss = vtrace(feature_screen_states, learner_select_unit_act_policies, learner_values, select_unit_act_policies, select_unit_act_actions, rewards, dones, select_unit_act_parametric_action_distribution)
        select_unit_id_args_loss = vtrace(feature_screen_states, learner_select_unit_id_policies, learner_values, select_unit_id_policies, select_unit_id_actions, rewards, dones, select_unit_id_parametric_action_distribution)
        select_worker_args_loss = vtrace(feature_screen_states, learner_select_worker_policies, learner_values, select_worker_policies, select_worker_actions, rewards, dones, select_worker_parametric_action_distribution)
        build_queue_id_args_loss = vtrace(feature_screen_states, learner_build_queue_id_policies, learner_values, build_queue_id_policies, build_queue_id_actions, rewards, dones, build_queue_parametric_action_distribution)
        unload_id_args_loss = vtrace(feature_screen_states, learner_unload_id_policies, learner_values, unload_id_policies, unload_id_actions, rewards, dones, unload_id_parametric_action_distribution)
        
        total_loss = fn_loss + screen_args_loss + minimap_args_loss + screen2_args_loss + queued_args_loss + control_group_act_args_loss + control_group_id_args_loss + \
                     select_point_act_args_loss + select_add_args_loss + select_unit_act_args_loss + select_unit_id_args_loss + select_worker_args_loss + build_queue_id_args_loss + \
                     unload_id_args_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads_norm = tf.linalg.global_norm(grads)
    grads, _ = tf.clip_by_global_norm(grads, arguments.gradient_clipping)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, grads_norm


def mask_unavailable_actions(available_actions, fn_pi):
    available_actions = tf.cast(available_actions, 'float32')
    fn_pi *= available_actions
    return fn_pi


def sample(logits):
    dist = tfd.Categorical(logits=logits)
    return dist.sample()


def sample_actions(available_actions, fn_pi, arg_pis):
  def sample(probs):
    dist = tfd.Categorical(logits=probs)
    return dist.sample()

  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi)

  return fn_samples, arg_samples



@tf.function
def prediction(feature_screen_state, feature_minimap_state, player_state, feature_units_state, available_actions_state, game_loop_state, build_queue_state, 
               single_select_state, multi_select_state, score_cumulative_state, act_history_state, memory_state, carry_state):
    prediction = model([feature_screen_state, feature_minimap_state, player_state, feature_units_state, game_loop_state, available_actions_state, build_queue_state,
                        single_select_state, multi_select_state, score_cumulative_state, act_history_state, memory_state, carry_state], training=False)

    fn_pi = prediction[0]

    fn_pi = tf.nn.softmax(fn_pi)
    fn_pi = mask_unavailable_actions(available_actions_state, fn_pi)
    fn_probs = fn_pi / tf.reduce_sum(fn_pi, axis=1, keepdims=True)
    fn_dist = tfd.Categorical(probs=fn_probs)
    fn_samples = fn_dist.sample()[0]

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

    memory_state = prediction[15]
    carry_state = prediction[16]

    return fn_samples, \
           screen_arg_samples, minimap_arg_samples, screen2_arg_samples, queued_arg_samples, control_group_act_arg_samples, control_group_id_arg_samples, \
           select_point_act_arg_samples, select_add_arg_samples, select_unit_act_arg_samples, select_unit_id_arg_samples, select_worker_arg_samples, \
           build_queue_id_arg_samples, unload_id_arg_samples, \
           fn_pi, \
           prediction[1], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6], prediction[7], prediction[8], prediction[9], prediction[10], \
           prediction[11], prediction[12], prediction[13], \
           memory_state, carry_state


@tf.function
def enque_data(env_id, reward, done, 

               feature_screen_state, feature_minimap_state, player_state, feature_units_state, available_actions_state, 
               game_loop_state, build_queue_state, single_select_state, multi_select_state, score_cumulative_state, act_history_state,

               fn_policy, screen_policy, minimap_policy, screen2_policy, queued_policy, control_group_act_policy, control_group_id_policy,
               select_point_act_policy, select_add_policy, select_unit_act_policy, select_unit_id_policy, select_worker_policy,
               build_queue_id_policy, unload_id_policy,

               fn_action, screen_action, minimap_action, screen2_action, queued_action, control_group_act_action, control_group_id_action, select_point_act_action,
               select_add_action, select_unit_act_action, select_unit_id_action, select_worker_action, build_queue_id_action, unload_id_action,

               memory_state, carry_state):
    queue.enqueue((env_id, reward, done,

                   feature_screen_state, feature_minimap_state, player_state, feature_units_state, available_actions_state, game_loop_state, build_queue_state, 
                   single_select_state, multi_select_state, score_cumulative_state, act_history_state, 

                   fn_policy, screen_policy, minimap_policy, screen2_policy, queued_policy, control_group_act_policy, control_group_id_policy, select_point_act_policy, select_add_policy, 
                   select_unit_act_policy, select_unit_id_policy, select_worker_policy, build_queue_id_policy, unload_id_policy,
                   
                   fn_action, screen_action, minimap_action, screen2_action, queued_action, control_group_act_action, control_group_id_action, select_point_act_action, select_add_action, 
                   select_unit_act_action, select_unit_id_action, select_worker_action, build_queue_id_action, unload_id_action,
                   
                   memory_state, carry_state))



def Data_Thread(coord, i):
    env_ids = np.zeros((unroll_length + 1), dtype=np.int32)

    feature_screen_states = np.zeros((unroll_length + 1, *feature_screen_size), dtype=np.float32)
    feature_minimap_states = np.zeros((unroll_length + 1, *feature_minimap_size), dtype=np.float32)
    player_states = np.zeros((unroll_length + 1, player_size), dtype=np.float32)
    feature_units_states = np.zeros((unroll_length + 1, *feature_units_size), dtype=np.float32)
    available_actions_states = np.zeros((unroll_length + 1, available_actions_size), dtype=np.float32)
    game_loop_states = np.zeros((unroll_length + 1, game_loop_size), dtype=np.float32)
    build_queue_states = np.zeros((unroll_length + 1, build_queue_size), dtype=np.float32)
    single_select_states = np.zeros((unroll_length + 1, single_select_size), dtype=np.float32)
    multi_select_states = np.zeros((unroll_length + 1, multi_select_size), dtype=np.float32)
    score_cumulative_states = np.zeros((unroll_length + 1, score_cumulative_size), dtype=np.float32)
    act_history_states = np.zeros((unroll_length + 1, *act_history_size), dtype=np.float32)

    fn_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    screen_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    minimap_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    screen2_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    queued_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    control_group_act_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    control_group_id_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    select_point_act_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    select_add_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    select_unit_act_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    select_unit_id_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    select_worker_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    build_queue_id_actions = np.zeros((unroll_length + 1), dtype=np.int32)
    unload_id_actions = np.zeros((unroll_length + 1), dtype=np.int32)

    fn_policies = np.zeros((unroll_length + 1, _NUM_FUNCTIONS), dtype=np.float32)
    screen_policies = np.zeros((unroll_length + 1, screen_action_size), dtype=np.float32)
    minimap_policies = np.zeros((unroll_length + 1, minimap_action_size), dtype=np.float32)
    screen2_policies = np.zeros((unroll_length + 1, screen2_action_size), dtype=np.float32)
    queued_policies = np.zeros((unroll_length + 1, queued_action_size), dtype=np.float32)
    control_group_act_policies = np.zeros((unroll_length + 1, control_group_act_action_size), dtype=np.float32)
    control_group_id_policies = np.zeros((unroll_length + 1, control_group_id_action_size), dtype=np.float32)
    select_point_act_policies = np.zeros((unroll_length + 1, select_point_act_action_size), dtype=np.float32) 
    select_add_policies = np.zeros((unroll_length + 1, select_add_action_size), dtype=np.float32)
    select_unit_act_policies = np.zeros((unroll_length + 1, select_unit_act_action_size), dtype=np.float32)
    select_unit_id_policies = np.zeros((unroll_length + 1, select_unit_id_action_size), dtype=np.float32)
    select_worker_policies = np.zeros((unroll_length + 1, select_worker_action_size), dtype=np.float32)
    build_queue_id_policies = np.zeros((unroll_length + 1, build_queue_id_action_size), dtype=np.float32)
    unload_id_policies = np.zeros((unroll_length + 1, unload_id_action_size), dtype=np.float32)

    rewards = np.zeros((unroll_length + 1), dtype=np.float32)
    dones = np.zeros((unroll_length + 1), dtype=np.bool)
    memory_states = np.zeros((unroll_length + 1, 256), dtype=np.float32)
    carry_states = np.zeros((unroll_length + 1, 256), dtype=np.float32)

    memory_index = 0

    index = 0
    memory_state = np.zeros([1,256], dtype=np.float32)
    carry_state = np.zeros([1,256], dtype=np.float32)
    min_elapsed_time = 5.0

    reward_list = []

    max_average_reward = 20.0

    while not coord.should_stop(): 
        start = time.time()

        message = socket_list[i].recv_pyobj()

        if memory_index == unroll_length:
            enque_data(env_ids, rewards, dones, 

                       feature_screen_states, feature_minimap_states, player_states, feature_units_states, available_actions_states, game_loop_states, build_queue_states, 
                       single_select_states, multi_select_states, score_cumulative_states, act_history_states,

                       fn_policies, screen_policies, minimap_policies, screen2_policies, queued_policies, control_group_act_policies, control_group_id_policies,
                       select_point_act_policies, select_add_policies, select_unit_act_policies, select_unit_id_policies, select_worker_policies,
                       build_queue_id_policies, unload_id_policies,

                       fn_actions, screen_actions, minimap_actions, screen2_actions, queued_actions, control_group_act_actions, control_group_id_actions, select_point_act_actions,
                       select_add_actions, select_unit_act_actions, select_unit_id_actions, select_worker_actions, build_queue_id_actions, unload_id_actions,

                       memory_states, carry_states)

            env_ids[0] = env_ids[memory_index]

            feature_screen_states[0] = feature_screen_states[memory_index]
            feature_minimap_states[0] = feature_minimap_states[memory_index]
            player_states[0] = player_states[memory_index]
            feature_units_states[0] = feature_units_states[memory_index]
            available_actions_states[0] = available_actions_states[memory_index]
            game_loop_states[0] = game_loop_states[memory_index]
            build_queue_states[0] = build_queue_states[memory_index]
            single_select_states[0] = single_select_states[memory_index]
            multi_select_states[0] = multi_select_states[memory_index]
            score_cumulative_states[0] = score_cumulative_states[memory_index]
            act_history_states[0] = act_history_states[memory_index]

            fn_actions[0] = fn_actions[memory_index]
            screen_actions[0] = screen_actions[memory_index]
            minimap_actions[0] = minimap_actions[memory_index]
            screen2_actions[0] = screen2_actions[memory_index]
            queued_actions[0] = queued_actions[memory_index]
            control_group_act_actions[0] = control_group_act_actions[memory_index]
            control_group_id_actions[0] = control_group_id_actions[memory_index]
            select_point_act_actions[0] = select_point_act_actions[memory_index]
            select_add_actions[0] = select_add_actions[memory_index]
            select_unit_act_actions[0] = select_unit_act_actions[memory_index]
            select_unit_id_actions[0] = select_unit_id_actions[memory_index]
            select_worker_actions[0] = select_worker_actions[memory_index]
            build_queue_id_actions[0] = build_queue_id_actions[memory_index]
            unload_id_actions[0] = unload_id_actions[memory_index]

            fn_policies[0] = fn_policies[memory_index]
            screen_policies[0] = screen_policies[memory_index]
            minimap_policies[0] = minimap_policies[memory_index]
            screen2_policies[0] = screen2_policies[memory_index]
            queued_policies[0] = queued_policies[memory_index]
            control_group_act_policies[0] = control_group_act_policies[memory_index]
            control_group_id_policies[0] = control_group_id_policies[memory_index]
            select_point_act_policies[0] = select_point_act_policies[memory_index]
            select_add_policies[0] = select_add_policies[memory_index]
            select_unit_act_policies[0] = select_unit_act_policies[memory_index]
            select_unit_id_policies[0] = select_unit_id_policies[memory_index]
            select_worker_policies[0] = select_worker_policies[memory_index]
            build_queue_id_policies[0] = build_queue_id_policies[memory_index]
            unload_id_policies[0] = unload_id_policies[memory_index]

            rewards[0] = rewards[memory_index]
            dones[0] = dones[memory_index]
            memory_states[0] = memory_states[memory_index]
            carry_states[0] = carry_states[memory_index]

            memory_index = 1

        feature_screen_state = tf.constant(np.array(message["feature_screen"]))
        feature_minimap_state = tf.constant(np.array(message["feature_minimap"]))
        player_state = tf.constant(np.array(message["player"]))
        feature_units_state = tf.constant(np.array(message["feature_units"]))
        available_actions_state = tf.constant(np.array(message["available_actions"]))
        game_loop_state = tf.constant(np.array(message["game_loop"]))
        build_queue_state = tf.constant(np.array(message["build_queue"]))
        single_select_state = tf.constant(np.array(message["single_select"]))
        multi_select_state = tf.constant(np.array(message["multi_select"]))
        score_cumulative_state = tf.constant(np.array(message["score_cumulative"]))
        act_history_state = tf.constant(np.array(message["act_history"]))

        prediction_return = prediction(feature_screen_state, feature_minimap_state, player_state, feature_units_state, available_actions_state, game_loop_state, 
                                       build_queue_state, single_select_state, multi_select_state, score_cumulative_state, act_history_state,
                                       memory_state, carry_state)
        env_ids[memory_index] = message["env_id"]

        feature_screen_states[memory_index] = message["feature_screen"]
        feature_minimap_states[memory_index] = message["feature_minimap"]
        player_states[memory_index] = message["player"]
        feature_units_states[memory_index] = message["feature_units"]
        available_actions_states[memory_index] = message["available_actions"]
        game_loop_states[memory_index] = message["game_loop"]
        build_queue_states[memory_index] = message["build_queue"]
        single_select_states[memory_index] = message["single_select"]
        multi_select_states[memory_index] = message["multi_select"]
        score_cumulative_states[memory_index] = message["score_cumulative"]
        act_history_states[memory_index] = message["act_history"]

        fn_action = prediction_return[0]
        screen_action = prediction_return[1]
        minimap_action= prediction_return[2]
        screen2_action= prediction_return[3]
        queued_action= prediction_return[4]
        control_group_act_action = prediction_return[5]
        control_group_id_action = prediction_return[6]
        select_point_act_action = prediction_return[7]
        select_add_action = prediction_return[8]
        select_unit_act_action = prediction_return[9]
        select_unit_id_action = prediction_return[10]
        select_worker_action= prediction_return[11]
        build_queue_id_action = prediction_return[12]
        unload_id_action = prediction_return[13]

        unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(fn_action)].args)
        for arg_type in unused_types:
            if arg_type.name == "screen":
                screen_action = -1
            elif arg_type.name == "minimap":
                minimap_action = -1
            elif arg_type.name == "screen2":
                screen2_action = -1
            elif arg_type.name == "queued":
                queued_action = -1
            elif arg_type.name == "control_group_act":
                control_group_act_action = -1
            elif arg_type.name == "control_group_id":
                control_group_id_action = -1
            elif arg_type.name == "select_point_act":
                select_point_act_action = -1
            elif arg_type.name == "select_add":
                select_add_action = -1
            elif arg_type.name == "select_unit_act":
                select_unit_act_action = -1
            elif arg_type.name == "select_unit_id":
                select_unit_id_action = -1
            elif arg_type.name == "select_worker":
                select_worker_action = -1
            elif arg_type.name == "build_queue_id":
                build_queue_id_action = -1
            elif arg_type.name == "unload_id":
                unload_id_action = -1
        
        fn_actions[memory_index] = fn_action
        screen_actions[memory_index] = screen_action
        minimap_actions[memory_index] = minimap_action
        screen2_actions[memory_index] = screen2_action
        queued_actions[memory_index] = queued_action
        control_group_act_actions[memory_index] = control_group_act_action
        control_group_id_actions[memory_index] = control_group_id_action
        select_point_act_actions[memory_index] = select_point_act_action
        select_add_actions[memory_index] = select_add_action
        select_unit_act_actions[memory_index] = select_unit_act_action
        select_unit_id_actions[memory_index] = select_unit_id_action
        select_worker_actions[memory_index] = select_worker_action
        build_queue_id_actions[memory_index] = build_queue_id_action
        unload_id_actions[memory_index] = unload_id_action

        fn_policies[memory_index] = prediction_return[14]
        screen_policies[memory_index] = prediction_return[15]
        minimap_policies[memory_index] = prediction_return[16]
        screen2_policies[memory_index] = prediction_return[17]
        queued_policies[memory_index] = prediction_return[18]
        control_group_act_policies[memory_index] = prediction_return[19]
        control_group_id_policies[memory_index] = prediction_return[20]
        select_point_act_policies[memory_index] = prediction_return[21]
        select_add_policies[memory_index] = prediction_return[22]
        select_unit_act_policies[memory_index] = prediction_return[23]
        select_unit_id_policies[memory_index] = prediction_return[24]
        select_worker_policies[memory_index] = prediction_return[25]
        build_queue_id_policies[memory_index] = prediction_return[26]
        unload_id_policies[memory_index] = prediction_return[27]

        rewards[memory_index] = message["reward"]
        dones[memory_index] = message["done"]
        memory_states[memory_index] = prediction_return[28]
        carry_states[memory_index] = prediction_return[29]

        reward_list.append(message["reward"])

        memory_state = prediction_return[28]
        carry_state = prediction_return[29]
 
        socket_list[i].send_pyobj({"env_id": message["env_id"], 

                                   "fn_action": fn_action, "screen_action": screen_action, "minimap_action": minimap_action, "screen2_action": screen2_action, 
                                   "queued_action": queued_action, "control_group_act_action": control_group_act_action, "control_group_id_action": control_group_id_action, 
                                   "select_point_act_action": select_point_act_action, "select_add_action": select_add_action, "select_unit_act_action": select_unit_act_action,
                                   "select_unit_id_action": select_unit_id_action, "select_worker_action": select_worker_action, "build_queue_id_action": build_queue_id_action, 
                                   "unload_id_action": unload_id_action})

        memory_index += 1
        index += 1

        #average_reward = sum(reward_list[-50:]) / len(reward_list[-50:])
        average_reward = message["average_reward"]
        if i == 0:
            print("average_reward: ", average_reward)
            if average_reward >= max_average_reward + 10:
                max_average_reward = average_reward
                model.save_weights('model/reinforcement_model_' + str(index))
                #print("average_reward: ", average_reward)

        end = time.time()
        elapsed_time = end - start

    if index == 100000000:
        coord.request_stop()


unroll_queues = []
unroll_queues.append(queue)

def dequeue(ctx):
    dequeue_outputs = tf.nest.map_structure(
        lambda *args: tf.stack(args), 
        *[unroll_queues[ctx].dequeue() for i in range(batch_size)]
      )

    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and repack.
    return tf.nest.flatten(dequeue_outputs)


def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(_dequeue, num_parallel_calls=1)


if arguments.gpu_use == True:
    device_name = '/device:GPU:0'
else:
    device_name = '/device:CPU:0'

dataset = dataset_fn(0)
it = iter(dataset)

@tf.function
def minimize(iterator):
    dequeue_data = next(iterator)
    total_loss, grads_norm = update(dequeue_data[3], dequeue_data[4], dequeue_data[5], dequeue_data[6], dequeue_data[7], dequeue_data[8], dequeue_data[9], dequeue_data[10], 
                                    dequeue_data[11], dequeue_data[12], dequeue_data[13],

                                    dequeue_data[28], dequeue_data[29], dequeue_data[30], dequeue_data[31], dequeue_data[32], dequeue_data[33], dequeue_data[34], dequeue_data[35], 
                                    dequeue_data[36], dequeue_data[37], dequeue_data[38], dequeue_data[39], dequeue_data[40], dequeue_data[41],

                                    dequeue_data[14], dequeue_data[15], dequeue_data[16], dequeue_data[17], dequeue_data[18], dequeue_data[19], dequeue_data[20], dequeue_data[21], 
                                    dequeue_data[22], dequeue_data[23], dequeue_data[24], dequeue_data[25], dequeue_data[26], dequeue_data[27],

                                    dequeue_data[1], dequeue_data[2], dequeue_data[42], dequeue_data[43])

    return total_loss, grads_norm


def Train_Thread(coord):
    index = 0

    while not coord.should_stop():
        #print("index : ", index)
        index += 1

        total_loss, grads_norm = minimize(it)
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=index)
            tf.summary.scalar("grads_norm", grads_norm, step=index)
            tf.summary.scalar("gradient_clipping", arguments.gradient_clipping, step=index)
            writer.flush()

        if index == 1000000000:
            coord.request_stop()


coord = tf.train.Coordinator(clean_stop_exception_types=None)

thread_data_list = []
for i in range(arguments.env_num):
    thread_data = threading.Thread(target=Data_Thread, args=(coord,i))
    thread_data_list.append(thread_data)

thread_train = threading.Thread(target=Train_Thread, args=(coord,))
thread_train.start()

for thread_data in thread_data_list:
    thread_data.start()

for thread_data in thread_data_list:
    coord.join(thread_data)

coord.join(thread_train)
