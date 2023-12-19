from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.actions import TYPES as ACTION_TYPES

import collections
import zmq
import gym
import numpy as np
import statistics
import tqdm
import glob
import random
import tensorflow as tf
import argparse
import os
from typing import Any, List, Sequence, Tuple
from absl import flags
import time
import utils

FLAGS = flags.FLAGS
FLAGS(['actor.py'])

parser = argparse.ArgumentParser(description='PySC2 IMPALA Actor')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
parser.add_argument('--environment', type=str, default='MoveToBeacon', 
                    choices=["MoveToBeacon", "CollectMineralShards", "FindAndDefeatZerglings",
                             "DefeatRoaches", "DefeatZerglingsAndBanelings"],
                    help='name of SC2 environment')
arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if arguments.env_id == 0:
    writer = tf.summary.create_file_writer("tensorboard_actor")

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))


is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type.name] = name in ['minimap', 'screen', 'screen2']

feature_screen_size = 16
feature_minimap_size = 16
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
use_raw_units = False
step_mul = 8
game_steps_per_episode = None
disable_fog = False
visualize = False

env_name = arguments.environment
players = [sc2_env.Agent(sc2_env.Race['terran'])]

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


def mask_unused_argument_samples(fn_id, arg_ids):
    args_out = dict()
    for arg_type in actions.TYPES:
        args_out[arg_type.name] = int(arg_ids[arg_type.name])

    a_0 = fn_id
    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
    for arg_type in unused_types:
        args_out[arg_type.name] = -1

    return fn_id, args_out


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


scores = []
episodes = []
average = []
average_reward = 0

for episode_step in range(0, 2000000):
    state = env.reset()

    feature_screen_history = np.zeros(((feature_screen_size, feature_screen_size, 6*4)))
    act_history = np.zeros((16, utils._NUM_FUNCTIONS))

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0

    if arguments.env_id == 0:
        start = time.time()

    while True:
        try:
            step += 1

            feature_screen = state[0][3]['feature_screen']
            feature_screen = utils.preprocess_screen(feature_screen)
            feature_screen = np.transpose(feature_screen, (1, 2, 0))
            feature_screen_history = np.roll(feature_screen_history, 6, axis=2)
            feature_screen_history[:,:,0:6] = feature_screen

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

            last_actions = state[0][3]['last_actions']
            if len(last_actions) != 0:
              last_actions_decoded = utils.preprocess_available_actions(last_actions[0])
            else:
              last_actions_decoded = utils.preprocess_available_actions(0)

            act_history = np.roll(act_history, 1, axis=0)
            act_history[0,:] = last_actions_decoded

            feature_screen_array = np.array([feature_screen_history])
            feature_minimap_array = np.array([feature_minimap])
            player_array = np.array([player])
            feature_units_array = np.array([feature_units])
            available_actions_array = np.array([available_actions])
            game_loop_array = np.array([game_loop])
            build_queue_array = np.array([build_queue])
            single_select_array = np.array([single_select])
            multi_select_array = np.array([multi_select])
            score_cumulative_array = np.array([score_cumulative])
            act_history_array = np.array([act_history])

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "feature_screen": feature_screen_array,
                          "feature_minimap": feature_minimap_array,
                          "player": player_array,
                          "feature_units": feature_units_array,
                          "available_actions": available_actions_array,
                          "game_loop": game_loop_array,
                          "build_queue": build_queue_array,
                          "single_select": single_select_array,
                          "multi_select": multi_select_array,
                          "score_cumulative": score_cumulative_array,
                          "act_history": act_history_array,
                          "average_reward": average_reward
                          }
            socket.send_pyobj(env_output)

            recv_pyobj = socket.recv_pyobj()

            fn_samples = int(recv_pyobj['fn_action'])
            screen_samples = int(recv_pyobj['screen_action'])
            minimap_samples = int(recv_pyobj['minimap_action'])
            screen2_samples = int(recv_pyobj['screen2_action'])
            queued_samples = int(recv_pyobj['queued_action'])
            control_group_act_samples = int(recv_pyobj['control_group_act_action'])
            control_group_id_samples = int(recv_pyobj['control_group_id_action'])
            select_point_act_samples = int(recv_pyobj['select_point_act_action'])
            select_add_samples = int(recv_pyobj['select_add_action'])
            select_unit_act_samples = int(recv_pyobj['select_unit_act_action'])
            select_unit_id_samples = int(recv_pyobj['select_unit_id_action'])
            select_worker_samples = int(recv_pyobj['select_worker_action'])
            build_queue_id_samples = int(recv_pyobj['build_queue_id_action'])
            unload_id_samples = int(recv_pyobj['unload_id_action'])

            arg_samples = dict()
            arg_samples['screen'] = screen_samples
            arg_samples['minimap'] = minimap_samples
            arg_samples['screen2'] = screen2_samples
            arg_samples['queued'] = queued_samples
            arg_samples['control_group_act'] = control_group_act_samples
            arg_samples['control_group_id'] = control_group_id_samples
            arg_samples['select_point_act'] = select_point_act_samples
            arg_samples['select_add'] = select_add_samples
            arg_samples['select_unit_act'] = select_unit_act_samples
            arg_samples['select_unit_id'] = select_unit_id_samples
            arg_samples['select_worker'] = select_worker_samples
            arg_samples['build_queue_id'] = build_queue_id_samples
            arg_samples['unload_id'] = unload_id_samples

            #fn_id, arg_ids = mask_unused_argument_samples(fn_samples, arg_samples)
            actions_list = actions_to_pysc2(fn_samples, arg_samples, (feature_screen_size, feature_screen_size))

            next_state = env.step(actions_list)
            done = next_state[0][0]
            if done == StepType.LAST:
                done = True
            else:
                done = False

            reward = float(next_state[0][1])

            if arguments.env_id == 0: 
                pass

            reward_sum += reward
            state = next_state
            if done:
                if arguments.env_id == 0:
                    end = time.time()
                    eposide_elasped_time = end - start
                    #print("eposide_elasped_time: ", eposide_elasped_time)

                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    average_reward = average[-1]
                    with writer.as_default():
                        #print("average[-1]: ", average[-1])
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    print("average_reward: " + str(average[-1]))
                else:
                    print("reward_sum: " + str(reward_sum))

                break

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')

env.close()
