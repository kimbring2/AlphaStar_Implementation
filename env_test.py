from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
import sys
import units_new
import upgrades_new

from utils import get_model_input, get_action_from_prediction, action_len, action_type_list, action_id_list
from network import EntityEncoder, SpatialEncoder, Core, ActionTypeHead, SelectedUnitsHead, TargetUnitHead, ScreenLocationHead, MinimapLocationHead
from trajectory import Trajectory

import random
import time
import math
import statistics
import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_name = 'Simple128'
players = [sc2_env.Agent(sc2_env.Race['terran']), 
            sc2_env.Agent(sc2_env.Race['terran'])]

feature_screen_size = 256
feature_minimap_size = 128
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
step_mul = 8
game_steps_per_episode = None
disable_fog = True
visualize = False

env = sc2_env.SC2Env(
      map_name=map_name,
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

env.reset()

class Agent(object):
  """Demonstrates agent interface.

  In practice, this needs to be instantiated with the right neural network
  architecture.
  """
  def __init__(self, race='Terran'):
    self.home_race = race
    self.away_race = 'Terran'

    self.build_order = []
    self.supply_depot_built = False

    self.scv_selected = False
    self.scv_return = False

    self.train_marine_flag = False
    self.train_marauder_flag = False

    self.build_supply_depot_flag = False
    self.build_barracks_flag = False
    self.build_refinery_flag = False
    self.build_techlab_flag = False

    self.marine_selected = False
    self.marauder_selected = False
    self.army_selected = False

    self.first_attack = False
    self.second_attack = False

    self.action_phase = 0
    self.previous_action = None
    self.selected_unit = []

    self.agent_model = None

    self.home_upgrade_array = np.zeros(89)
    self.away_upgrade_array = np.zeros(89)
  
  def make_model(self):
      feature_minimap = tf.keras.Input(shape=[11, 128, 128])
      embedded_feature_units = tf.keras.Input(shape=[512,464])
      core_prev_state = (tf.keras.Input(shape=[128]), tf.keras.Input(shape=[128]))
      embedded_scalar = tf.keras.Input(shape=[307])
      scalar_context = tf.keras.Input(shape=[842])

      map_, embedded_spatial = SpatialEncoder(img_height=128, img_width=128, channel=11)(feature_minimap)
      embedded_entity, entity_embeddings = EntityEncoder(464, 8)(embedded_feature_units)
      lstm_output, final_memory_state, final_carry_state = Core(128)(core_prev_state, embedded_entity, embedded_spatial, embedded_scalar)
      action_type_logits, action_type, autoregressive_embedding_action = ActionTypeHead(action_len)(lstm_output, scalar_context)
      selected_units_logits, selected_units, autoregressive_embedding_select = SelectedUnitsHead()(autoregressive_embedding_action, 
                                                                                                                         action_type, 
                                                                                                                         entity_embeddings)
      target_unit_logits, target_unit = TargetUnitHead()(autoregressive_embedding_select, action_type, entity_embeddings)

      screen_target_location_logits, screen_target_location = ScreenLocationHead()(autoregressive_embedding_select, action_type, map_)
      minimap_target_location_logits, minimap_target_location = MinimapLocationHead()(autoregressive_embedding_select, action_type, map_)
      agent_model = tf.keras.Model(
          inputs=[feature_minimap, embedded_feature_units, core_prev_state, embedded_scalar, scalar_context],
          outputs=[action_type_logits, action_type, selected_units_logits, selected_units, target_unit_logits, target_unit, 
                     screen_target_location_logits, screen_target_location, minimap_target_location_logits, minimap_target_location,
                     final_memory_state, final_carry_state, autoregressive_embedding_action]
      )
      
      self.agent_model = agent_model
  
  def step(self, observation, core_state):
    feature_minimap, embedded_feature_units, embedded_scalar, scalar_context = get_model_input(self, observation)

    feature_minimap_list = []
    embedded_feature_units_list = []
    core_state_array_0 = []
    core_state_array_1 = []
    embedded_scalar_list=  []
    scalar_context_list = []

    batch_size = 1
    for i in range(0, batch_size):
      feature_minimap_list.append(feature_minimap)
      embedded_feature_units_list.append(embedded_feature_units)
      core_state_array_0.append(core_state[0])
      core_state_array_1.append(core_state[1])
      embedded_scalar_list.append(embedded_scalar)
      scalar_context_list.append(scalar_context)

    feature_minimap_array  = np.vstack(feature_minimap_list)
    embedded_feature_units_array = np.vstack(embedded_feature_units_list)
    core_state_array = (np.vstack(core_state_array_0), np.vstack(core_state_array_1))  
    embedded_scalar_array = np.vstack(embedded_scalar_list)
    scalar_context_array = np.vstack(scalar_context_list)

    predict_value = self.agent_model([feature_minimap_array, embedded_feature_units_array, core_state_array, 
                                             embedded_scalar_array, scalar_context_array])

    #print("predict_value: " + str(predict_value))
    action_type_logits = predict_value[0]
    action_type = predict_value[1]
    selected_units_logits = predict_value[2]
    selected_units = predict_value[3]
    target_unit_logits = predict_value[4]
    target_unit = predict_value[5]
    screen_target_location_logits = predict_value[6]
    screen_target_location_x = predict_value[7][0]
    screen_target_location_y = predict_value[7][1]
    minimap_target_location_logits = predict_value[8]
    minimap_target_location_x = predict_value[9][0]
    minimap_target_location_y = predict_value[9][1]
    final_memory_state = predict_value[10]
    final_carry_state = predict_value[11]
    
    #print("action_type_logits.shape: " + str(action_type_logits.shape))
    #print("action_type: " + str(action_type))
    #print("selected_units_logits.shape: " + str(selected_units_logits.shape))
    #print("selected_units: " + str(selected_units))
    #print("target_unit_logits.shape: " + str(target_unit_logits.shape))
    #print("target_unit: " + str(target_unit))
    #print("target_location_logits.shape: " + str(target_location_logits.shape))
    #print("screen_target_location_x: " + str(screen_target_location_x))
    #print("screen_target_location_y: " + str(screen_target_location_y))
    #print("minimap_target_location_x: " + str(minimap_target_location_x))
    #print("minimap_target_location_y: " + str(minimap_target_location_y))
    #print("final_memory_state.shape: " + str(final_memory_state.shape))
    #print("final_carry_state.shape: " + str(final_carry_state.shape))
    #print("")

    core_new_state = (final_memory_state, final_carry_state)
    
    action_ = get_action_from_prediction(self, observation, 
                                                action_type.numpy(), selected_units.numpy(), target_unit.numpy(), 
                                                screen_target_location_x.numpy(), screen_target_location_y.numpy(),
                                                minimap_target_location_x.numpy(), minimap_target_location_y.numpy())
    
    action = [action_, action_type, selected_units, target_unit, 
                screen_target_location_x, screen_target_location_y, minimap_target_location_x, minimap_target_location_y]
    policy_logits = [action_type_logits, selected_units_logits, target_unit_logits, screen_target_location_logits, minimap_target_location_logits]
    new_state = core_new_state
    
    return action, policy_logits, new_state


agent1 = Agent(race='Terran')
agent1.make_model()

agent2 = Agent()
'''
obs = env.reset()
core_prev_state = (np.zeros([1,128]), np.zeros([1,128]))
for i in range(0, 1000):
  print("i: " + str(i))
  # action_1 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]

  action_1, policy_logits_1, new_state_1 = agent1.step(obs[0][3], core_prev_state)
  #print("new_state_1[0].shape: " + str(new_state_1[0].shape))
  #print("new_state_1[1].shape: " + str(new_state_1[1].shape))
  #print("action_1: " + str(action_1))
  core_prev_state = new_state_1

  #action_2, policy_logits_2, new_state_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]
  print("action_1[0][0]: " + str(action_1[0][0]))
  obs = env.step([action_1[0][0], action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  print("")
'''
replay = Trajectory('/media/kimbring2/Steam/StarCraftII/Replays/', 'Terran', 'Terran', 2500)
replay.get_random_trajectory()

replay_index = 0
core_prev_state = (np.zeros([1,128]), np.zeros([1,128]))
scce = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.001)
writer = tf.summary.create_file_writer("/media/kimbring2/Steam/AlphaStar_Implementation/tfboard")
for p in range(0, 1000):
  print("replay_index: " + str(replay_index))

  online_variables = agent1.agent_model.trainable_variables
  with tf.GradientTape() as tape:
    tape.watch(online_variables)

    feature_screen_list = []
    embedded_feature_units_list = []
    core_state0_list = []
    core_state1_list = []
    embedded_scalar_list = [] 
    scalar_context_list = []
    acts_human_list = []
    acts_agent_list = []
    for i in range(replay_index, replay_index + 4):
      trajectory = replay.home_trajectory[i][0]
      acts_human = replay.home_trajectory[i][1]
      #print("acts_human: " + str(acts_human))
      action_1, policy_logits_1, new_state_1 = agent1.step(trajectory, core_prev_state)
      acts_agent_list.append(action_1)
      acts_human_list.append(acts_human)
      core_prev_state = new_state_1

      feature_screen, embedded_feature_units, embedded_scalar, scalar_context = get_model_input(agent1, trajectory)
      feature_screen_list.append(feature_screen)
      embedded_feature_units_list.append(embedded_feature_units)
      core_state0_list.append(core_prev_state[0])
      core_state1_list.append(core_prev_state[1])
      embedded_scalar_list.append(embedded_scalar)
      scalar_context_list.append(scalar_context)

    feature_screen_array  = np.vstack(feature_screen_list)
    embedded_feature_units_array = np.vstack(embedded_feature_units_list)
    core_state_array = (np.vstack(core_state0_list), np.vstack(core_state1_list))  
    embedded_scalar_array = np.vstack(embedded_scalar_list)
    scalar_context_array = np.vstack(scalar_context_list)

    predict_value = agent1.agent_model([feature_screen_array, embedded_feature_units_array, core_state_array, 
                                                embedded_scalar_array, scalar_context_array])
    all_losses = 0 
    for i in range(0, 4):
      acts_human = acts_human_list[i]
      #print("acts_human: " + str(acts_human))

      for act_human in acts_human:
        #print("act_human: " + str(act_human))
        human_function = str(act_human.function)
        if int(act_human.function) == 2 or int(act_human.function) == 3 or int(act_human.function) == 4 or int(act_human.function) == 5 \
            or int(act_human.function) == 6:
          # _Functions.select_control_group, 
          human_action_with_argument = [int(act_human.function), int(act_human.arguments[0][0])]
          human_action_index = action_id_list.index(human_action_with_argument)
        else:
          human_arguments = str(act_human.arguments)
          human_action_name = human_function.split('.')[-1]
          human_action_index = action_id_list.index([int(actions._Functions[human_action_name])])

        if human_action_index != predict_value[1][i].numpy():
          action_true = [human_action_index]
          action_pred = predict_value[0][i]

          action_loss = scce(action_true, action_pred)
          all_losses += 0.5 *action_loss
        else:
          print("act_human: " + str(act_human))
          print("act_human.arguments: " + str(act_human.arguments))
          '''
          outputs=[action_type_logits, action_type, selected_units_logits, selected_units, target_unit_logits, target_unit, 
                     screen_target_location_logits, screen_target_location, minimap_target_location_logits, minimap_target_location,
                     final_memory_state, final_carry_state, autoregressive_embedding_action]
          '''
          selected_units_pred = predict_value[2][i]
          target_unit_pred = predict_value[4][i]
          screen_target_location_pred = predict_value[6][i]
          minimap_target_location_pred = predict_value[8][i]
          
          print("selected_units_pred.shape: " + str(selected_units_pred.shape))
          print("target_unit_pred.shape: " + str(target_unit_pred.shape))
          print("screen_target_location_pred.shape: " + str(screen_target_location_pred.shape))
          print("minimap_target_location_pred.shape: " + str(minimap_target_location_pred.shape))

          arg_list = action_type_list[human_action_index][0].args
          print("arg_list: " + str(arg_list))
          for i, arg in enumerate(arg_list):
            print("arg.id: " + str(arg.id))
            #print("arg[i]: " + str(arg[i]))
            human_arg = act_human.arguments[i]
            print("human_arg: " + str(human_arg))
            
            if arg.id == 0:
              # action_type_arg.name: screen
              # action_type_arg.sizes: (0, 0)human_arg[0] * 256 + human_arg[1]
              screen_position_human = [int(human_arg[0]) * 256 + int(human_arg[1])]
              screen_position_agent = [screen_target_location_pred]
              screen_loss = scce(screen_position_human, screen_position_agent)
              all_losses += 0.2 * screen_loss
            elif arg.id == 1:
              # action_type_arg.name: minimap
              # action_type_arg.sizes: (0, 0)

              #print("human_arg[0]: " + str(human_arg[0]))
              #print("human_arg[1]: " + str(human_arg[1]))
              minimap_position_human = [int(human_arg[0]) * 128 + int(human_arg[1])]
              minimap_position_agent = [screen_target_location_pred]
              minimap_loss = scce(minimap_position_human, minimap_position_agent)
              all_losses += 0.2 * minimap_loss
            elif arg.id == 2:
              # action_type_arg.name: screen2
              # action_type_arg.sizes: (0, 0)
              screen_position_human = [int(human_arg[0]) * 256 + int(human_arg[1])]
              screen_position_agent = [screen_target_location_pred]
              screen_loss = scce(screen_position_human, screen_position_agent)
              all_losses += 0.2 * screen_loss
            elif arg.id == 3:
              pass
              # action_type_arg.name: queued
              # action_type_arg.sizes: (2,)
              #act_name = 'now'
              #if act_name == 'now':
              #  argument.append([0])
              #elif act_name == 'queued':
              #  argument.append([1])
            elif arg.id == 4:
              # action_type_arg.name: control_group_act
              # action_type_arg.sizes: (5,)
              pass
            elif arg.id == 5:
              # action_type_arg.name: control_group_id
              # action_type_arg.sizes: (10,)
              control_group_id_human = [int(human_arg)]
              control_group_id_agent = [selected_units_pred]

              control_group_id_loss = scce(control_group_id_human, control_group_id_agent)
              all_losses += 0.2 * control_group_id_loss
            elif arg.id == 6:
              # action_type_arg.name: select_point_act
              # action_type_arg.sizes: (4,)
              pass
            elif arg.id == 7:
              # action_type_arg.name: select_add
              # action_type_arg.sizes: (2,)
              pass
            elif arg.id == 8:
              # action_type_arg.name: select_unit_act
              # action_type_arg.sizes: (4,)
              pass
            elif arg.id == 9:
              # action_type_arg.name: select_unit_id
              # action_type_arg.sizes: (500,)
              selected_unit_id_human = [int(human_arg)]
              selected_unit_id_agent = [selected_units_pred]

              selected_unit_id_loss = scce(selected_unit_id_human, selected_unit_id_agent)
              all_losses += 0.2 * selected_unit_id_loss
            elif arg.id == 10:
              # action_type_arg.name: select_worker
              # action_type_arg.sizes: (4,)
              pass
            elif arg.id == 11:
              # action_type_arg.name: build_queue_id
              # action_type_arg.sizes: (10,)
              build_queue_id_human = [int(human_arg)]
              build_queue_id_agent = [selected_units_pred]

              build_queue_id_loss = scce(build_queue_id_human, build_queue_id_agent)
              all_losses += 0.2 * build_queue_id_loss
              argument.append(selected_units)
            elif arg.id == 12:
              # action_type_arg.name: unload_id
              # action_type_arg.sizes: (500,)
              unload_id_human = [int(human_arg)]
              unload_id_agent = [selected_units_pred]

              unload_id_loss = scce(unload_id_human, unload_id_agent)
              all_losses += 0.2 * unload_id_loss
          

          '''
          policy_logits = [action_type_logits, selected_units_logits, target_unit_logits, target_location_logits]
          '''
          #print("acts_agent_list[i][1]: " + str(acts_agent_list[i][1]))
          #print("acts_agent_list[i][2]: " + str(acts_agent_list[i][2]))
          #print("acts_agent_list[i][3]: " + str(acts_agent_list[i][3]))
          #print("acts_agent_list[i][4]: " + str(acts_agent_list[i][4]))
          #print("acts_agent_list[i][5]: " + str(acts_agent_list[i][5]))

    if all_losses != 0:
      print("all_losses: " + str(all_losses))
      #tf.summary.scalar('all_losses', all_losses, step=replay_index)
      gradients = tape.gradient(all_losses, online_variables)
      optimizer.apply_gradients(zip(gradients, online_variables))

    print("")
    replay_index += 1
    if replay_index == len(replay.home_trajectory):
        print("Replay end")
        break
