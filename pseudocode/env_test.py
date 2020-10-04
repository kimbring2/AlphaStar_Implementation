from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
import sys
import units_new
import upgrades_new

from utils import get_entity_obs, get_upgrade_obs, get_gameloop_obs, get_race_onehot, get_agent_statistics
from network import EntityEncoder, ScalarEncoder, SpatialEncoder, Core

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

env.save_replay("rulebase_replay")

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

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
    #self.race = race
    #self.steps = 0
    #self.weights = initial_weights
    self.supply_depot_built = False

    self.scv_selected = False
    self.scv_return = False

    self.marine_selected = False
    self.marauder_selected = False
    self.army_selected = False

    self.first_attack = False
    self.second_attack = False

    self.spatial_encoder = SpatialEncoder(img_height=128, img_width=128, channel=27)
    self.scalar_encoder = ScalarEncoder(128)
    self.entity_encoder = EntityEncoder(464, 8)
    self.core = Core(12)

    self.previous_action = None


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
    # feature_screen.shape: (27, 128, 128)
    feature_minimap = observation[3]['feature_minimap']
    feature_units = observation[3]['feature_units']
    feature_player = observation[3]['player']
    score_by_category = observation[3]['score_by_category']
    game_loop = observation[3]['game_loop']

    spatial_encoder_output = self.spatial_encoder(np.reshape(feature_screen, [1,128,128,27]))
    #print("spatial_encoder.shape: " + str(spatial_encoder_output.shape))

    agent_statistics = get_agent_statistics(score_by_category)
    # agent_statistics.shape: (55,)

    home_race = 'Terran'
    away_race = 'Terran'
    race = get_race_onehot(home_race, away_race)
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

    available_actions = observation[3]['available_actions']
    cumulative_statistics = observation[3]['score_cumulative']
    #print("available_actions: " + str(available_actions))
    #print("cumulative_statistics: " + str(cumulative_statistics))
    if (self.previous_action is not None):
      previous_action = (self.previous_action)
      #print("previous_action: " + str(previous_action))
      if previous_action == _BUILD_SUPPLY_DEPOT:
        print("_BUILD_SUPPLY_DEPOT true")
        self.previous_action = None

      #_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
      #_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
      #_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
      #_BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

      #_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
      #_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
      #_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
      #print("previous_action: " + str(previous_action))
      # beginning_build_order = 

    #scalar_context = 
    # embedded_scalar.shape: (307,)

    scalar_encoder_output = self.scalar_encoder(np.reshape(embedded_scalar, [1,307]))
    #print("scalar_encoder_output.shape: " + str(scalar_encoder_output.shape))

    embedded_feature_units = get_entity_obs(feature_units)
    # embedded_feature_units.shape: (512, 464)
    entity_encoder_output = self.entity_encoder(np.reshape(embedded_feature_units, [1,512,464]))
    #print("entity_encoder_output.shape: " + str(entity_encoder_output.shape))

    # Encoder Input
    #spatial_encoder.shape: (1, 16384)
    #scalar_encoder_output.shape: (1, 128)
    #entity_encoder_output.shape: (1, 256)
    encoder_input = np.concatenate((spatial_encoder_output, scalar_encoder_output, entity_encoder_output), axis=1)
    # encoder_input.shape: (1, 16768)

    core_input = np.reshape(encoder_input, [16, 8, 131])
    whole_seq_output, final_memory_state, final_carry_state = self.core(core_input)
    #print(whole_seq_output.shape)
    #print(final_memory_state.shape)
    #print(final_carry_state.shape)

    available_actions = observation[3]['available_actions']

    unit_type = feature_screen.unit_type
    empty_space = np.where(unit_type == 0)
    empty_space = np.vstack((empty_space[0], empty_space[1])).T
    #print("feature_minimap: " + str(feature_minimap))

    enermy = (feature_minimap.player_relative == _PLAYER_ENEMY).nonzero()
    enermy = np.vstack((enermy[0], enermy[1])).T

    self_CommandCenter = [unit for unit in feature_units if unit.unit_type == units.Terran.CommandCenter and unit.alliance == _PLAYER_SELF]
    self_SupplyDepot = [unit for unit in feature_units if unit.unit_type == units.Terran.SupplyDepot and unit.alliance == _PLAYER_SELF]
    self_Refinery = [unit for unit in feature_units if unit.unit_type == units.Terran.Refinery and unit.alliance == _PLAYER_SELF]
    self_BarracksTechLab = [unit for unit in feature_units if unit.unit_type == units.Terran.BarracksTechLab and unit.alliance == _PLAYER_SELF]
    self_SCVs = [unit for unit in feature_units if unit.unit_type == units.Terran.SCV and unit.alliance == _PLAYER_SELF]
    self_Marines = [unit for unit in feature_units if unit.unit_type == units.Terran.Marine and unit.alliance == _PLAYER_SELF]
    self_Marauder = [unit for unit in feature_units if unit.unit_type == units.Terran.Marauder and unit.alliance == _PLAYER_SELF]
    self_Barracks = [unit for unit in feature_units if unit.unit_type == units.Terran.Barracks and unit.alliance == _PLAYER_SELF]
    self_BarracksFlying = [unit for unit in feature_units if unit.unit_type == units.Terran.BarracksFlying and unit.alliance == _PLAYER_SELF]
    neutral_Minerals = [unit for unit in feature_units if unit.unit_type == units.Neutral.MineralField]
    neutral_VespeneGeysers = [unit for unit in feature_units if unit.unit_type == units.Neutral.VespeneGeyser]

    unselected_SCV_list = []
    for SCV in self_SCVs:
      if SCV.is_selected == 0:
        unselected_SCV_list.append(SCV)
    
    num_SCV = len(self_SCVs)
    num_Barracks = len(self_Barracks)
    num_BarracksFlying = len(self_BarracksFlying)
    num_Marines = len(self_Marines)
    num_Marauder= len(self_Marauder)
    num_Refinery = len(self_Refinery)
    num_BarracksTechLab = len(self_BarracksTechLab)

    total_Barracks = num_Barracks + num_BarracksFlying + num_BarracksTechLab
    #print("first_attack: " + str(first_attack))
    if num_Refinery != 0:
      assigned_scv = self_Refinery[0].assigned_harvesters

    x_list = []
    y_list = []
    for Mineral in neutral_Minerals:
      x_list.append(Mineral.x)
      y_list.append(Mineral.y)

    num_Minerals = len(x_list)
    if (num_Minerals != 0):
      mean_x_Minerals = statistics.mean(x_list)
      mean_y_Minerals = statistics.mean(y_list)
      dis_x = self_CommandCenter[0].x - mean_x_Minerals
      dis_y = self_CommandCenter[0].y - mean_y_Minerals
    
    self_minerals = feature_player.minerals
    self_vespene = feature_player.vespene
    self_food_used = feature_player.food_used
    self_food_cap = feature_player.food_cap

    #print("first_attack: " + str(first_attack))
    action = [actions.FUNCTIONS.no_op()]
    if (self.first_attack == False):
      if (self_food_cap - self_food_used <=3):
        if (self.scv_selected == False):
          self.scv_selected = True
          random_SCV = random.choice(unselected_SCV_list)
          target = [random_SCV.x, random_SCV.y]
          action = [actions.FUNCTIONS.select_point("select", target)]
        else:
          if ( (self_minerals >= 100) & (_BUILD_SUPPLY_DEPOT in available_actions) ):
            #target = [self_CommandCenter[0].x + dis_x, self_CommandCenter[0].y + dis_y]
            random_point = random.choice(empty_space)
            target = [random_point[0], random_point[1]]
            #print("target: " + str(target))

            self.previous_action = _BUILD_SUPPLY_DEPOT
            action = [actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])]
      elif (num_Barracks <= 2):
        if ( (self_minerals >= 150) & (_BUILD_BARRACKS in available_actions) ):
            #target = [self_CommandCenter[0].x + dis_x, self_CommandCenter[0].y + dis_y]
            random_point = random.choice(empty_space)
            target = [random_point[0], random_point[1]]
            #print("target: " + str(target))

            self.previous_action = _BUILD_BARRACKS
            action = [actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])]
      elif (num_Marines < 10):
        if (num_Barracks != 0):
          if ( (self_minerals >= 50) & (_TRAIN_MARINE in available_actions) ):

              self.previous_action = _TRAIN_MARINE
              action = [actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])]
          else:
            self.scv_selected = False
            random_barrack = random.choice(self_Barracks)
            target = [random_barrack.x, random_barrack.y]
            action = [actions.FUNCTIONS.select_point("select", target)]
      elif (num_Marines >= 10):
          if ( (_SELECT_ARMY in available_actions) & (self.marine_selected == False) ) :
            self.marine_selected = True
            action = [actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])]
          elif (self.marine_selected == True):
            if (_ATTACK_MINIMAP in available_actions):
              self.first_attack = True
              random_point = random.choice(enermy)
              target = [random_point[0], random_point[1]]
              action = [actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])]
      else:
        if (_HARVEST_GATHER in available_actions) :
          target = [neutral_Minerals[0].x, neutral_Minerals[0].y]
          action = [actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED, target])]
        else:
          if (_SELECT_IDLE_WORKER in available_actions):
            action = [actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])]
    else:
      if (self_food_cap - self_food_used <=3):
        #print("scv_selected: " + str(scv_selected))

        if (self.scv_selected == False):
          self.scv_selected = True
          random_SCV = random.choice(unselected_SCV_list)
          target = [random_SCV.x, random_SCV.y]
          action = [actions.FUNCTIONS.select_point("select", target)]
        else:
          if ( (self_minerals >= 100) & (_BUILD_SUPPLY_DEPOT in available_actions) ):
            #target = [self_CommandCenter[0].x + dis_x, self_CommandCenter[0].y + dis_y]
            random_point = random.choice(empty_space)
            target = [random_point[0], random_point[1]]
            #print("target: " + str(target))

            self.previous_action = _BUILD_SUPPLY_DEPOT
            action = [actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])]
      elif (num_Refinery == 0):
        if (_BUILD_REFINERY not in available_actions):
          random_SCV = random.choice(unselected_SCV_list)
          target = [random_SCV.x, random_SCV.y]
          action = [actions.FUNCTIONS.select_point("select", target)]
        elif ( (self_minerals >= 100) & (_BUILD_REFINERY in available_actions) ):
          #target = [random_point[0], random_point[1]]
          #print("target: " + str(target))
          if (len(neutral_VespeneGeysers) != 0):
            target = [neutral_VespeneGeysers[0].x, neutral_VespeneGeysers[0].y]

            self.previous_action = _BUILD_REFINERY
            action = [actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target])]
      elif (self_Refinery[0].assigned_harvesters <= 3):
        #print("self_Refinery[0].assigned_harvesters: " + str(self_Refinery[0].assigned_harvesters))
        #print("Refinery loop")
        if (self.scv_selected == False):
          #print("scv_selected command")
          self.scv_selected = True
          random_SCV = random.choice(unselected_SCV_list)
          target = [random_SCV.x, random_SCV.y]
          action = [actions.FUNCTIONS.select_point("select", target)]
          #print("select scv")
        elif ( (_HARVEST_GATHER in available_actions) & (self.scv_selected == True) ):
          #print("havest command")
          self.scv_selected = False
          target = [self_Refinery[0].x, self_Refinery[0].y]
          action = [actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED, target])]
      elif (num_SCV <= 14):
        if (_TRAIN_SCV not in available_actions):
          if (len(self_CommandCenter) != 0):
            target = [self_CommandCenter[0].x, self_CommandCenter[0].y]
            action = [actions.FUNCTIONS.select_point("select", target)]
            self.scv_selected = False
        else:
          action = [actions.FunctionCall(_TRAIN_SCV, [_QUEUED])]
      elif (num_Marines < 10):
        if (num_Barracks != 0):
          if ( (self_minerals >= 50) & (_TRAIN_MARINE in available_actions) ):

              self.previous_action = _TRAIN_MARINE
              action = [actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])]
          else:
            target = [self_Barracks[0].x, self_Barracks[0].y]
            action = [actions.FUNCTIONS.select_point("select", target)]
      elif (total_Barracks <= 2):
        if ( (self_minerals >= 150) & (_BUILD_BARRACKS in available_actions) ):
          #target = [self_CommandCenter[0].x + dis_x, self_CommandCenter[0].y + dis_y]
          random_point = random.choice(empty_space)
          target = [random_point[0], random_point[1]]
          #print("target: " + str(target))

          self.previous_action = _BUILD_BARRACKS
          action = [actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])]
        else:
          random_SCV = random.choice(unselected_SCV_list)
          target = [random_SCV.x, random_SCV.y]
          action = [actions.FUNCTIONS.select_point("select", target)]
      elif (num_BarracksTechLab == 0):
        #print("TechLab loop")
        #print("num_BarracksTechLab: " + str(num_BarracksTechLab))

        if (_BUILD_TECHLAB not in available_actions):
          #target = [self_Barracks[0].x, self_Barracks[0].y]
          #random_point = random.choice(empty_space)
          #target = [random_point[0], random_point[1]]
          target = [self_Barracks[0].x, self_Barracks[0].y]
          action = [actions.FUNCTIONS.select_point("select", target)]
        elif ( (self_minerals >= 100) & (self_vespene >= 25) & (_BUILD_TECHLAB in available_actions) ):
          #target = [self_Barracks[0].x, self_Barracks[0].y]
          random_point = random.choice(empty_space)
          target = [random_point[0], random_point[1]]

          self.previous_action = _BUILD_TECHLAB
          action = [actions.FunctionCall(_BUILD_TECHLAB, [_NOT_QUEUED, target])]
      elif (num_Marauder < 5):
        #print("Marauder loop")
        if (num_Barracks != 0):
          if ( (self_minerals >= 100) & (self_vespene >= 25) & (_TRAIN_MARAUDER in available_actions) ):
              #print("_TRAIN_MARAUDER command")

              self.previous_action = _TRAIN_MARAUDER
              action = [actions.FunctionCall(_TRAIN_MARAUDER, [_QUEUED])]
          else:
            self.scv_selected = False
            target = [self_Barracks[1].x, self_Barracks[1].y]
            action = [actions.FUNCTIONS.select_point("select", target)]
      elif (num_Marauder >= 3):
        if ( (_SELECT_ARMY in available_actions) & (self.marauder_selected == False) ):
          self.marauder_selected = True
          self.scv_selected = False
          action = [actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])]
        elif ( (_SELECT_CONTROL_GROUP in available_actions) & (self.marauder_selected == True) ):
          #marauder_selected = False
          #random_point = random.choice(enermy)
          #target = [random_point[0], random_point[1]]
          action = [actions.FUNCTIONS.select_control_group("set", 1)]
          #action = [actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])]
      else:
        if ( (_SELECT_CONTROL_GROUP in available_actions) & (self.army_selected == False) ):
          self.army_selected = True
          action = [actions.FUNCTIONS.select_control_group("recall", 1)]
        elif ( (_ATTACK_MINIMAP in available_actions) & (self.army_selected == True) ):
          random_point = random.choice(enermy)
          target = [random_point[0], random_point[1]]
          action = [actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])]

    return action


agent1 = Agent()
agent2 = Agent()

obs = env.reset()
#for i in range(0, 20000):
while True:
  #print("action: " + str(action))
  #print("num_Marauder: " + str(num_Marauder))

  action_1 = agent1.step(obs[0])
  #action_1 = [actions.FUNCTIONS.no_op()]
  #print("action_1: " + str(action_1))

  #action_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op()]
  obs = env.step([action_1, action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  #print("")