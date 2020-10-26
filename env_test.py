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
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_RETURN_SCV = actions.FUNCTIONS.Harvest_Return_SCV_quick.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_HARVEST_GATHER_SCV = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id

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

    ################################################################################################
    self.spatial_encoder = SpatialEncoder(img_height=128, img_width=128, channel=27)
    self.entity_encoder = EntityEncoder(464, 8)

    self.core = Core(256)

    self.action_type_head = ActionTypeHead(7)
    self.selected_units_head = SelectedUnitsHead()
    self.target_unit_head = TargetUnitHead()
    self.location_head = LocationHead()
    ################################################################################################

    self.core_prev_state = None
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
      action_acceptable_entity_type_binary = tf.keras.Input(shape=[512])

      map_, embedded_spatial = SpatialEncoder(img_height=128, img_width=128, channel=27)(feature_screen)
      embedded_entity, entity_embeddings = EntityEncoder(464, 8)(embedded_feature_units)
      
      whole_seq_output, final_memory_state, final_carry_state = Core(256)(core_prev_state, embedded_entity, embedded_spatial, embedded_scalar)
      lstm_output = tf.reshape(whole_seq_output, [1, 9 * 256])
      
      action_type_logits, action_type, autoregressive_embedding = ActionTypeHead(7)(lstm_output, scalar_context)
      selected_units_logits_, selected_units_, autoregressive_embedding = SelectedUnitsHead()(autoregressive_embedding, 
                                                                                                                   action_acceptable_entity_type_binary, 
                                                                                                                   entity_embeddings)
      target_unit_logits, target_unit = TargetUnitHead()(autoregressive_embedding, action_acceptable_entity_type_binary, entity_embeddings)
      target_location_logits, target_location = self.location_head(autoregressive_embedding, action_type, map_)

    
      # Instantiate an end-to-end model predicting both priority and department
      agent_model = tf.keras.Model(
          inputs=[feature_screen, embedded_feature_units, core_prev_state, embedded_scalar, scalar_context, action_acceptable_entity_type_binary],
          outputs=[action_type_logits, selected_units_logits_, target_unit_logits, target_location_logits],
      )

      agent_model.summary()

      self.agent_model = agent_model
  
  def step(self, observation):
    global home_upgrade_array
    global away_upgrade_array
    global previous_action

    """Performs inference on the observation, given hidden state last_state."""
    # We are omitting the details of network inference here.
    # ...
    #print("observation: " + str(observation))
    #print("")
    feature_screen = observation[3]['feature_screen']
    #print("feature_screen.shape: " + str(feature_screen.shape))
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
    #print("time.shape: " + str(time.shape))
    #time.shape : (64,)

    upgrade_value = get_upgrade_obs(feature_units)
    if upgrade_value != -1:
      home_upgrade_array[np.where(upgrade_value[0] == 1)] = 1
      away_upgrade_array[np.where(upgrade_value[1] == 1)] = 1

    # home_upgrade_array.shape: (89,)
    # away_upgrade_array.shape: (89,)

    embedded_scalar = np.concatenate((agent_statistics, race, time, home_upgrade_array, away_upgrade_array), axis=0)
    embedded_scalar = np.expand_dims(embedded_scalar, axis=0)
    #print("embedded_scalar.shape: " + str(embedded_scalar.shape))

    cumulative_statistics = observation[3]['score_cumulative'] / 1000.0
    # cumulative_statistics.: [1050    2    0  600  400    0    0    0    0    0    0    0    0]

    cumulative_statistics_array = np.log(cumulative_statistics + 1)
    #print("cumulative_statistics_array.shape: " + str(cumulative_statistics_array.shape))

    build_order_array = np.zeros(256)
    if (self.previous_action is not None):
      previous_action = (self.previous_action)
      #print("previous_action: " + str(previous_action))

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
        #print("unit_info: " + str(unit_info))
        build_order_array[unit_info] = 1

        if len(self.build_order) <= 20:
          self.build_order.append(build_order_array)

        unit_name = None

    feature_screen = np.expand_dims(feature_screen, axis=0)
    map_, embedded_spatial = self.spatial_encoder(feature_screen)

    available_actions_array = np.zeros(573)
    available_actions_list = available_actions.tolist()
    for available_action in available_actions_list:
      available_actions_array[available_action] = 1

    #print("available_actions_array.shape: " + str(available_actions_array.shape))
    
    # available_actions_array.shape: (573,)
    # cumulative_statistics.shape: (13,)
    # build_order_array.shape: (256,)
    scalar_context = np.concatenate((available_actions_array, cumulative_statistics_array, build_order_array), axis=0)
    scalar_context = np.reshape(scalar_context, [1, 842])
    # scalar_context.shape: (1, 842)

    embedded_feature_units = get_entity_obs(feature_units)
    embedded_feature_units = np.reshape(embedded_feature_units, [1,512,464])
    #print("embedded_feature_units.shape: " + str(embedded_feature_units.shape))
    embedded_entity, entity_embeddings = self.entity_encoder(embedded_feature_units)
    action = [actions.FUNCTIONS.no_op()]
    #print("entity_embeddings.shape: " + str(entity_embeddings.shape))
    #print("embedded_entity.shape: " + str(embedded_entity.shape))
    
    #print("embedded_spatial.shape: " + str(embedded_spatial.shape))
    #print("embedded_scalar.shape: " + str(embedded_scalar.shape))
    #print("embedded_entity.shape: " + str(embedded_entity.shape))
    # embedded_spatial.shape: (1, 256)
    # embedded_scalar.shape: (1, 307)
    # embedded_entity.shape: (1, 256)
    whole_seq_output, final_memory_state, final_carry_state = self.core(self.core_prev_state, embedded_entity, embedded_spatial, embedded_scalar)
    #print("whole_seq_output.shape: " + str(whole_seq_output.shape))
    # whole_seq_output.shape: (1, 9, 256)

    #print("final_memory_state.shape: " + str(final_memory_state.shape))
    # final_memory_state.shape: (1, 256)
    self.core_prev_state = (final_memory_state, final_carry_state)
    # self.core_prev_state[0].shape: (1, 256)
    # self.core_prev_state[1].shape: (1, 256)

    lstm_output = np.reshape(whole_seq_output, [1, 9 * 256])

    action_type_list = [_BUILD_SUPPLY_DEPOT, _BUILD_BARRACKS, _BUILD_REFINERY, _TRAIN_MARINE, _TRAIN_MARAUDER, _ATTACK_MINIMAP, _BUILD_TECHLAB]
    action = [actions.FUNCTIONS.no_op()]

    action_type_logits, action_type, autoregressive_embedding = self.action_type_head(lstm_output, scalar_context) 
    #print("action_type: " + str(action_type))
    
    selectable_entity_mask = np.zeros(512)
    for idx, feature_unit in enumerate(feature_units):
        #print("feature_unit: " + str(feature_unit))
        selectable_entity_mask[idx] = 1

    action_acceptable_entity_type_binary = np.zeros(512)
    if action_type == 0 or action_type == 1 or action_type == 2:
      action_acceptable_entity_type_binary[43] = 1 
    elif action_type == 3 or action_type == 4:
      action_acceptable_entity_type_binary[3] = 1 
    elif action_type == 5:
      action_acceptable_entity_type_binary[28] = 1 
      action_acceptable_entity_type_binary[29] = 1
      action_acceptable_entity_type_binary[43] = 1  
    elif action_type == 6:
      action_acceptable_entity_type_binary[3] = 1 

    action_acceptable_entity_type_binary = np.expand_dims(action_acceptable_entity_type_binary, 0)
    #print("action_acceptable_entity_type_binary.shape: " + str(action_acceptable_entity_type_binary.shape))
    selected_units_logits_, selected_units_, autoregressive_embedding = self.selected_units_head(autoregressive_embedding, 
                                                                                                                       action_acceptable_entity_type_binary, 
                                                                                                                       entity_embeddings)
    #print("feature_units: " + str(feature_units))
    #print("len(feature_units): " + str(len(feature_units)))
    #print("selected_units_.numpy(): " + str(selected_units_.numpy()))
    #print("feature_units[selected_units_.numpy()]: " + str(feature_units[selected_units_.numpy()]))
    if (selected_units_.numpy() < len(feature_units)):
      selected_units_ = selected_units_.numpy()
      self.selected_unit.append(feature_units[selected_units_])
      #print("feature_units[selected_units_]: " + str(feature_units[selected_units_]))
    else:
      selected_units_ = None

    target_unit_logits, target_unit = self.target_unit_head(autoregressive_embedding, action_acceptable_entity_type_binary, entity_embeddings)
    if (target_unit < len(feature_units)):
      target_unit = target_unit
      #print("feature_units[target_unit]: " + str(feature_units[target_unit]))
    else:
      target_unit = None

    target_location_logits, target_location = self.location_head(autoregressive_embedding, action_type, map_)
    #print("target_location: " + str(target_location))
    if self.action_phase == 0 and selected_units_ is not None and (_SELECT_POINT in available_actions):
      #selected_units = random.choice(self_SCVs) # 2. Selected Units Head
      selected_units = feature_units[selected_units_]
      #print("selected_units: " + str(selected_units))

      select_point = [selected_units.x, selected_units.y]
      action = [actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, select_point])]
      self.action_phase = 1
    elif self.action_phase == 1 and action_type_list[action_type] in available_actions:
      #target_unit = None # 3. Target Unit Head
      target_unit = target_unit

      #position = random.choice(empty_space) # 4. Location Head
      position = (target_location[0], target_location[1])
      action = [actions.FunctionCall(action_type_list[action_type], [_NOT_QUEUED, position])]

    self.previous_action = action 
    
    policy_logits = None
    new_state = None

    action_type_, selected_units_, target_unit_, target_location_ = self.agent_model([feature_screen, embedded_feature_units, 
                                                                                                        self.core_prev_state, 
                                                                                                        embedded_scalar, scalar_context, action_acceptable_entity_type_binary])
    #result = self.agent_model([feature_screen, embedded_feature_units, self.core_prev_state, embedded_scalar])
    print("action_type_: " + str(action_type_))
    print("selected_units_: " + str(selected_units_))
    print("target_unit_: " + str(target_unit_))
    print("target_location_: " + str(target_location_))

    return action, policy_logits, new_state


agent1 = Agent()
agent1.make_model()

agent2 = Agent()

obs = env.reset()
#for i in range(0, 20000):
while True:
  #print("action: " + str(action))
  #print("num_Marauder: " + str(num_Marauder))

  action_1, policy_logits_1, new_state_1 = agent1.step(obs[0])
  #action_1 = [actions.FUNCTIONS.no_op()]
  #print("action_1: " + str(action_1))

  #action_2, policy_logits_2, new_state_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op()]
  obs = env.step([action_1, action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  #print("")