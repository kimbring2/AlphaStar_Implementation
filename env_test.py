from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
import sys
import units_new
import upgrades_new

from utils import get_entity_obs, get_upgrade_obs, get_gameloop_obs, get_race_onehot, get_agent_statistics
from network import EntityEncoder, SpatialEncoder, Core, ActionTypeHead, SelectedUnitsHead, TargetUnitHead, LocationHead
from trajectory import Trajectory

import random
import time
import math
import statistics
import numpy as np
import tensorflow as tf

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_name = 'Simple128'
#players = [sc2_env.Agent(sc2_env.Race['terran']), 
#           sc2_env.Bot(sc2_env.Race['protoss'], sc2_env.Difficulty.very_easy)]
players = [sc2_env.Agent(sc2_env.Race['terran']), 
            sc2_env.Agent(sc2_env.Race['terran'])]

feature_screen_size = 128
feature_minimap_size = 64
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

#env.save_replay("rulebase_replay")

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

# Action part
_NO_OP = actions.FUNCTIONS.no_op.id

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_NOT_QUEUED = [0]
_QUEUED = [1]

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_SMART_MINIMAP = actions.FUNCTIONS.Smart_minimap.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_BUILD_COMMANDCENTER_SCREEN = actions.FUNCTIONS.Build_CommandCenter_screen.id
_BUILD_SUPPLYDEPOT_SCREEN = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS_SCREEN = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_TECHLAB_SCREEN = actions.FUNCTIONS.Build_TechLab_screen.id
_BUILD_TECHLAB_QUICK = actions.FUNCTIONS.Build_TechLab_quick.id
_BUILD_REACTOR_QUICK = actions.FUNCTIONS.Build_Reactor_quick.id
_BUILD_BUNKER_SCREEN = actions.Build_Bunker_screen.id
_BUILD_STARPORT_SCREEN = actions.Build_Starport_screen.id
_BUILD_FACTORY_SCREEN = actions.Build_Factory_screen.id

_TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER_QUICK = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_SCV_QUICK = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_SIEGETANK_QUICK = actions.FUNCTIONS.Train_SiegeTank_quick.id
_TRAIN_MEDIVAC_QUICK = actions.FUNCTIONS.Train_Medivac_quick.id
_TRAIN_REAPER_QUICK = actions.FUNCTIONS.Train_Reaper_quick

_RETURN_SCV_QUICK = actions.FUNCTIONS.Harvest_Return_SCV_quick.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_HARVEST_GATHER_SCV_SCREEN = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_LIFT_QUICK = actions.FUNCTIONS._Functions.Lift_quick.id
_MORPH_SUPPLYDEPOT_LOWER_QUICK = actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id
_LAND_SCREEN = actions.FUNCTIONS.Land_screen

home_upgrade_array = np.zeros(89)
away_upgrade_array = np.zeros(89)
class Agent(object):
  """Demonstrates agent interface.

  In practice, this needs to be instantiated with the right neural network
  architecture.
  """
  def __init__(self):
    self.home_race = 'Terran'
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

    self.core_prev_state = (tf.zeros([1, 256]), tf.zeros([1, 256]))
    self.action_phase = 0
    self.previous_action = None
    self.selected_unit = []

    self.agent_model = None
  
  def make_model(self):
      feature_screen = tf.keras.Input(shape=[27, 128, 128])
      embedded_feature_units = tf.keras.Input(shape=[512,464])
      core_prev_state = (tf.keras.Input(shape=[256]), tf.keras.Input(shape=[256]))
      embedded_scalar = tf.keras.Input(shape=[307])
      scalar_context = tf.keras.Input(shape=[842])
      #action_acceptable_entity_type_binary = tf.keras.Input(shape=[512])

      map_, embedded_spatial = SpatialEncoder(img_height=128, img_width=128, channel=27)(feature_screen)
      embedded_entity, entity_embeddings = EntityEncoder(464, 8)(embedded_feature_units)
      
      whole_seq_output, final_memory_state, final_carry_state = Core(256)(core_prev_state, embedded_entity, embedded_spatial, embedded_scalar)
      lstm_output = tf.reshape(whole_seq_output, [1, 9 * 256])
      
      action_type_logits, action_type, autoregressive_embedding = ActionTypeHead(7)(lstm_output, scalar_context)
      selected_units_logits_, selected_units_, autoregressive_embedding = SelectedUnitsHead()(autoregressive_embedding, 
                                                                                                                   action_type, 
                                                                                                                   entity_embeddings)
      target_unit_logits, target_unit = TargetUnitHead()(autoregressive_embedding, action_type, entity_embeddings)
      target_location_logits, target_location = LocationHead()(autoregressive_embedding, action_type, map_)

      # Instantiate an end-to-end model predicting both priority and department
      agent_model = tf.keras.Model(
          inputs=[feature_screen, embedded_feature_units, core_prev_state, embedded_scalar, scalar_context],
          outputs=[action_type_logits, action_type, selected_units_logits_, selected_units_, target_unit_logits, target_unit, target_location_logits, target_location, 
                     final_memory_state, final_carry_state]
      )

      agent_model.summary()

      self.agent_model = agent_model
  
  def step(self, observation):
    global home_upgrade_array
    global away_upgrade_array
    global previous_action

    """Performs inference on the observation, given hidden state last_state."""
    # We are omitting the details of network inference here.
    feature_screen = observation[3]['feature_screen']
    # feature_screen.shape: (27, 128, 128)

    feature_minimap = observation[3]['feature_minimap']
    feature_units = observation[3]['feature_units']
    feature_player = observation[3]['player']
    score_by_category = observation[3]['score_by_category']
    game_loop = observation[3]['game_loop']
    available_actions = observation[3]['available_actions']
    # available_actions: [  0   1   2   3   4 264  12  13 274 549 451 452 453 331 332 333 334  79]

    agent_statistics = get_agent_statistics(score_by_category)
    # agent_statistics.shape: (55,)

    race = get_race_onehot(self.home_race, self.away_race)
    # race.shape: (10,)

    time = get_gameloop_obs(game_loop)
    #time.shape : (64,)

    upgrade_value = get_upgrade_obs(feature_units)
    if upgrade_value != -1:
      home_upgrade_array[np.where(upgrade_value[0] == 1)] = 1
      away_upgrade_array[np.where(upgrade_value[1] == 1)] = 1

    # home_upgrade_array.shape: (89,)
    # away_upgrade_array.shape: (89,)

    embedded_scalar = np.concatenate((agent_statistics, race, time, home_upgrade_array, away_upgrade_array), axis=0)
    embedded_scalar = np.expand_dims(embedded_scalar, axis=0)

    cumulative_statistics = observation[3]['score_cumulative'] / 1000.0
    # cumulative_statistics.: [1050    2    0  600  400    0    0    0    0    0    0    0    0]

    cumulative_statistics_array = np.log(cumulative_statistics + 1)

    build_order_array = np.zeros(256)
    if (self.previous_action is not None):
      previous_action = (self.previous_action)

      unit_name = None
      if previous_action == _BUILD_SUPPLY_DEPOT:
        #print("_BUILD_SUPPLY_DEPOT true")
        unit_name = 'SupplyDepot'
      elif previous_action == _BUILD_BARRACKS:
        unit_name = 'Barracks'
      elif previous_action == _BUILD_REFINERY:
        unit_name = 'Refinery'
      elif previous_action == _BUILD_TECHLAB:
        unit_name = 'TechLab'
      elif previous_action == _TRAIN_SCV:
        unit_name = 'SCV'
      elif previous_action == _TRAIN_MARINE:
        unit_name = 'Marine'
      elif previous_action == _TRAIN_MARAUDER:
        unit_name = 'Marauder'

      self.previous_action = None
      if unit_name is not None:
        unit_info = int(units_new.get_unit_type(self.home_race, unit_name)[0])
        build_order_array[unit_info] = 1

        if len(self.build_order) <= 20:
          self.build_order.append(build_order_array)

        unit_name = None

    feature_screen = np.expand_dims(feature_screen, axis=0)

    available_actions_array = np.zeros(573)
    available_actions_list = available_actions.tolist()
    for available_action in available_actions_list:
      available_actions_array[available_action] = 1

    # available_actions_array.shape: (573,)
    # cumulative_statistics.shape: (13,)
    # build_order_array.shape: (256,)

    scalar_context = np.concatenate((available_actions_array, cumulative_statistics_array, build_order_array), axis=0)
    scalar_context = np.reshape(scalar_context, [1, 842])
    # scalar_context.shape: (1, 842)

    embedded_feature_units = get_entity_obs(feature_units)
    embedded_feature_units = np.reshape(embedded_feature_units, [1,512,464])
    action = [actions.FUNCTIONS.no_op()]
    # embedded_spatial.shape: (1, 256)
    # embedded_scalar.shape: (1, 307)
    # embedded_entity.shape: (1, 256)

    predict_value = self.agent_model([feature_screen, embedded_feature_units, self.core_prev_state, embedded_scalar, scalar_context])
    action_type_logits = predict_value[0]
    action_type = predict_value[1].numpy()
    selected_units_logits = predict_value[2]
    selected_units = predict_value[3].numpy()
    target_unit_logits = predict_value[4]
    target_unit = predict_value[5].numpy()
    target_location_logits = predict_value[6]
    target_location = predict_value[7]
    final_memory_state = predict_value[8]
    final_carry_state = predict_value[9]

    #print("action_type_logits: " + str(action_type_logits))
    #print("action_type: " + str(action_type))
    #print("selected_units_logits_: " + str(selected_units_logits_))
    #print("selected_units: " + str(selected_units))
    #print("target_unit_logits: " + str(target_unit_logits))
    #print("target_unit: " + str(target_unit))
    #print("target_location_logits: " + str(target_location_logits))
    #print("target_location[0]: " + str(target_location[0]))
    #print("target_location[1]: " + str(target_location[1]))
    #print("")
    #print("final_memory_state: " + str(final_memory_state))
    #print("final_carry_state: " + str(final_carry_state))

    #self.core_prev_state = (final_memory_state, final_carry_state)
    new_state = (final_memory_state, final_carry_state)
    # self.core_prev_state[0].shape: (1, 256)
    # self.core_prev_state[1].shape: (1, 256)

    # FunctionCall(function=<_Functions.no_op: 0>, arguments=[])


    # FunctionCall(function=<_Functions.Smart_screen: 451>, arguments=[[<Queued.now: 0>], [98, 76]])
    # FunctionCall(function=<_Functions.Smart_minimap: 452>, arguments=[[<Queued.now: 0>], [43, 44]])

    # FunctionCall(function=<_Functions.select_point: 2>, arguments=[[<SelectPointAct.select: 0>], [81, 63]])
    # FunctionCall(function=<_Functions.select_rect: 3>, arguments=[[<SelectAdd.select: 0>], [19, 26], [63, 58]])

    # FunctionCall(function=<_Functions.move_camera: 1>, arguments=[[12, 18]])
    
    # FunctionCall(function=<_Functions.select_control_group: 4>, arguments=[[<ControlGroupAct.recall: 0>], [1]])
    # FunctionCall(function=<_Functions.select_control_group: 4>, arguments=[[<ControlGroupAct.recall: 0>], [2]])

    # FunctionCall(function=<_Functions.Train_Marine_quick: 477>, arguments=[[<Queued.now: 0>]])
    # FunctionCall(function=<_Functions.Train_SiegeTank_quick: 492>, arguments=[[<Queued.now: 0>]])
    # FunctionCall(function=<_Functions.Train_Medivac_quick: 478>, arguments=[[<Queued.now: 0>]])

    # FunctionCall(function=<_Functions.Build_Bunker_screen: 43>, arguments=[[<Queued.now: 0>], [80, 53]])
    # FunctionCall(function=<_Functions.Build_Reactor_quick: 71>, arguments=[[<Queued.now: 0>]])
    # FunctionCall(function=<_Functions.Build_Factory_screen: 53>, arguments=[[<Queued.now: 0>], [92, 70]])
    # FunctionCall(function=<_Functions.Build_Barracks_screen: 42>, arguments=[[<Queued.now: 0>], [103, 64]])
    # FunctionCall(function=<_Functions.Build_Refinery_screen: 79>, arguments=[[<Queued.now: 0>], [25, 55]])

    # FunctionCall(function=<_Functions.Morph_SupplyDepot_Lower_quick: 318>, arguments=[[<Queued.now: 0>]])
    # FunctionCall(function=<_Functions.Morph_SiegeMode_quick: 317>, arguments=[[<Queued.now: 0>]])

    # FunctionCall(function=<_Functions.Attack_screen: 12>, arguments=[[<Queued.now: 0>], [49, 90]])
    # FunctionCall(function=<_Functions.Attack_minimap: 13>, arguments=[[<Queued.now: 0>], [17, 42]])
    # FunctionCall(function=<_Functions.Attack_minimap: 13>, arguments=[[<Queued.queued: 1>], [11, 21]])

    # FunctionCall(function=<_Functions.select_army: 7>, arguments=[[<SelectAdd.select: 0>]])

    action_type_list = [_BUILD_SUPPLYDEPOT_SCREEN, _BUILD_BARRACKS_SCREEN, _BUILD_REFINERY_SCREEN, _BUILD_TECHLAB_SCREEN, _BUILD_COMMANDCENTER_SCREEN, 
                            _BUILD_REACTOR_QUICK, _BUILD_BUNKER_SCREEN, _BUILD_STARPORT_SCREEN, _BUILD_FACTORY_SCREEN
                            _TRAIN_MARINE_QUICK, _TRAIN_MARAUDER_QUICK, _TRAIN_SCV_QUICK, _TRAIN_SIEGETANK_QUICK, _TRAIN_MEDIVAC_QUICK, _TRAIN_REAPER_QUICK,
                            _RETURN_SCV_QUICK, _HARVEST_GATHER_SCREEN, _HARVEST_GATHER_SCV_SCREEN, 
                            _SELECT_CONTROL_GROUP, _LIFT_QUICK, _MORPH_SUPPLYDEPOT_LOWER_QUICK, _LAND_SCREEN,
                            _ATTACK_SCREEN, _ATTACK_MINIMAP, _SMART_SCREEN, _SMART_MINIMAP, 
                            _SELECT_POINT, _SELECT_RECT, _SELECT_IDLE_WORKER, _SELECT_CONTROL_GROUP, _SELECT_ARMY,
                            _MOVE_SCREEN, _MOVE_CAMERA]
    action = [actions.FUNCTIONS.no_op()]

    selectable_entity_mask = np.zeros(512)
    for idx, feature_unit in enumerate(feature_units):
        selectable_entity_mask[idx] = 1

    if (selected_units < len(feature_units)):
      self.selected_unit.append(feature_units[selected_units])
    else:
      selected_units = None

    if (target_unit < len(feature_units)):
      target_unit = target_unit
    else:
      target_unit = None

    if self.action_phase == 0 and selected_units is not None and (_SELECT_POINT in available_actions):
      selected_units_info = feature_units[selected_units]

      select_point = [selected_units_info.x, selected_units_info.y]
      action = [actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, select_point])]
      self.action_phase = 1
    elif self.action_phase == 1 and action_type_list[action_type] in available_actions:
      target_unit = target_unit
      position = (target_location[0], target_location[1])
      action = [actions.FunctionCall(action_type_list[action_type], [_NOT_QUEUED, position])]

    self.previous_action = action 
    
    policy_logits = [action_type_logits, selected_units_logits, target_unit_logits, target_location_logits]
    new_state = self.core_prev_state

    return action, policy_logits, new_state


#replay = Trajectory('/media/kimbring2/Steam/StarCraftII/Replays/', 'Terran', 'Terran', 2500)
#replay.get_random_trajectory()

agent1 = Agent()
agent1.make_model()

agent2 = Agent()

obs = env.reset()

replay_index = 0
while True:
  #obs = [0, 0, 0, replay.home_trajectory[replay_index][0]]
  #act = replay.home_trajectory[replay_index][1]
  print("replay_index: " + str(replay_index))

  replay_index += 1
  #action_1, policy_logits_1, new_state_1 = agent1.step(obs[0])
  action_1, policy_logits_1, new_state_1 = agent1.step(obs[0])

  #action_1, policy_logits_1, new_state_1 = agent1.step(obs[0])
  #print("action_1: " + str(action_1))
  #print("policy_logits_1: " + str(policy_logits_1))
  #print("new_state_1: " + str(new_state_1))

  agent1.core_prev_state = new_state_1

  action_1 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]
  #print("action_1: " + str(action_1))

  #action_2, policy_logits_2, new_state_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op()]
  obs = env.step([action_1, action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  #print("")