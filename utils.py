from pysc2.lib import actions, features, units
import numpy as np
import math
from collections import namedtuple
import os
import random
import collections
import threading
import time
import timeit
from absl import logging
import tensorflow as tf

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_NUM_FUNCTIONS = len(actions.FUNCTIONS)

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCREEN_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_VISIBILITY_MAP = features.SCREEN_FEATURES.visibility_map.index

_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index

FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])
FLAT_FEATURES = [
  FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
  FlatFeature(1,  features.FeatureType.SCALAR, 1, 'minerals'),
  FlatFeature(2,  features.FeatureType.SCALAR, 1, 'vespene'),
  FlatFeature(3,  features.FeatureType.SCALAR, 1, 'food_used'),
  FlatFeature(4,  features.FeatureType.SCALAR, 1, 'food_cap'),
  FlatFeature(5,  features.FeatureType.SCALAR, 1, 'food_army'),
  FlatFeature(6,  features.FeatureType.SCALAR, 1, 'food_workers'),
  FlatFeature(7,  features.FeatureType.SCALAR, 1, 'idle_worker_count'),
  FlatFeature(8,  features.FeatureType.SCALAR, 1, 'army_count'),
  FlatFeature(9,  features.FeatureType.SCALAR, 1, 'warp_gate_count'),
  FlatFeature(10, features.FeatureType.SCALAR, 1, 'larva_count'),
]

unit_list = [0, 48, 317]
def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_UNIT_TYPE:
      scale = len(unit_list)
      layer = np.zeros([scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(scale):
        indy, indx = (screen[i] == unit_list[j]).nonzero()
        if indy.shape[0] != 0:
          #print("unit_list[j]: ", unit_list[j])
          layer[j, indy, indx] = 1
            
      layers.append(layer)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif i == _SCREEN_PLAYER_ID or i == _SCREEN_SELECTED or i == _SCREEN_VISIBILITY_MAP:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
        
      layers.append(layer)
    elif i == _SCREEN_UNIT_HIT_POINTS:
      layers.append(np.log(screen[i:i+1] + 1) / np.log(features.SCREEN_FEATURES[i].scale))

  return np.concatenate(layers, axis=0)


def preprocess_player(player):
  layers = []
  for s in FLAT_FEATURES:
    out = player[s.index] / s.scale
    layers.append(out)

  return layers


def preprocess_available_actions(available_action):
    available_actions = np.zeros(_NUM_FUNCTIONS, dtype=np.float32)
    available_actions[available_action] = 1

    return available_actions


def preprocess_feature_units(feature_units, feature_screen_size):
    feature_units_list = []
    feature_units_length = len(feature_units)
    for i, feature_unit in enumerate(feature_units):
      feature_unit_length = len(feature_unit) 

      feature_unit_list = []
      feature_unit_list.append(feature_unit.unit_type / 2000)
      feature_unit_list.append(feature_unit.alliance / 4)
      feature_unit_list.append(feature_unit.health / 10000)
      feature_unit_list.append(feature_unit.shield / 10000)
      feature_unit_list.append(feature_unit.x / 100)
      feature_unit_list.append(feature_unit.y / 100) 
      feature_unit_list.append(feature_unit.is_selected)

      #print("feature_unit.x / (feature_screen_size + 1): ", feature_unit.x / (feature_screen_size + 1))
      #print("feature_unit.y / (feature_screen_size + 1): ", feature_unit.y / (feature_screen_size + 1))

      feature_units_list.append(feature_unit_list)
      #print("i: ", i)
      if i >= 49:
        break

    if feature_units_length < 50:
      for i in range(feature_units_length, 50):
        feature_units_list.append(np.zeros(7))

    entity_array = np.array(feature_units_list)
    
    return entity_array


def get_model_input(agent, observation):
  feature_screen = observation['feature_screen']
  feature_minimap = observation['feature_minimap']
  feature_units = observation['feature_units']
  feature_player = observation['player']
  score_by_category = observation['score_by_category']
  game_loop = observation['game_loop']
  available_actions = observation['available_actions']

  agent_statistics = get_agent_statistics(score_by_category)
  race = get_race_onehot(agent.home_race, agent.away_race)

  #print("game_loop: " + str(game_loop))
  time = get_gameloop_obs(game_loop)
  #print("time.shape: " + str(time.shape))

  upgrade_value = get_upgrade_obs(feature_units)
  if upgrade_value != -1 and upgrade_value is not None :
    agent.home_upgrade_array[np.where(upgrade_value[0] == 1)] = 1
    agent.away_upgrade_array[np.where(upgrade_value[1] == 1)] = 1

  embedded_scalar = np.concatenate((agent_statistics, race, time, agent.home_upgrade_array, agent.away_upgrade_array), axis=0)
  embedded_scalar = np.expand_dims(embedded_scalar, axis=0)

  cumulative_statistics = observation['score_cumulative'] / 1000.0
  cumulative_statistics_array = np.log(cumulative_statistics + 1)

  build_order_array = np.zeros(256)
  if (agent.previous_action is not None):
    unit_name = None
    if agent.previous_action == _BUILD_BARRACKS_SCREEN:
      unit_name = 'Barracks'
    elif agent.previous_action == _BUILD_REFINERY_SCREEN:
      unit_name = 'Refinery'
    elif agent.previous_action == _BUILD_TECHLAB_SCREEN or agent.previous_action == _BUILD_TECHLAB_QUICK:
      unit_name = 'TechLab'
    elif agent.previous_action == _BUILD_COMMANDCENTER_SCREEN:
      unit_name = 'CommandCenter'
    elif agent.previous_action == _BUILD_REACTOR_SCREEN or agent.previous_action == _BUILD_REACTOR_QUICK:
      unit_name = 'Reactor'
    elif agent.previous_action == _BUILD_BUNKER_SCREEN:
      unit_name = 'Bunker'
    elif agent.previous_action == _BUILD_STARPORT_SCREEN:
      unit_name = 'Starport'
    elif agent.previous_action == _BUILD_FACTORY_SCREEN:
      unit_name = 'Factory'
    elif agent.previous_action == _BUILD_ARMORY_SCREEN:
      unit_name = 'Armory'
    elif agent.previous_action == _BUILD_ENGINNERINGBAY_SCREEN:
      unit_name = 'EngineeringBay'
    elif agent.previous_action == _TRAIN_MARINE_QUICK:
      unit_name = 'Marine'
    elif agent.previous_action == _TRAIN_MARAUDER_QUICK:
      unit_name = 'Marauder'
    elif agent.previous_action == _TRAIN_SIEGETANK_QUICK:
      unit_name = 'SiegeTank'
    elif agent.previous_action == _TRAIN_MEDIVAC_QUICK:
      unit_name = 'Medivac'
    elif agent.previous_action == _TRAIN_REAPER_QUICK:
      unit_name = 'Reaper'
    elif agent.previous_action == _TRAIN_HELLION_QUICK:
      unit_name = 'Hellion'
    elif agent.previous_action == _TRAIN_VIKINGFIGHTER_QUICK:
      unit_name = 'VikingFighter'

    if unit_name is not None:
      unit_info = int(units_new.get_unit_type(agent.home_race, unit_name)[0])
      build_order_array[unit_info] = 1

      if len(agent.build_order) <= 20:
        agent.build_order.append(build_order_array)

      unit_name = None

  feature_minimap = np.expand_dims(feature_minimap, axis=0)
  available_actions_array = np.zeros(573)
  available_actions_list = available_actions.tolist()
  for available_action in available_actions_list:
    available_actions_array[available_action] = 1

  scalar_context = np.concatenate((available_actions_array, cumulative_statistics_array, build_order_array), axis=0)
  scalar_context = np.reshape(scalar_context, [1, 842])

  embedded_feature_units = get_entity_obs(feature_units)
  embedded_feature_units = np.reshape(embedded_feature_units, [1,512,464])

  return feature_minimap, embedded_feature_units, embedded_scalar, scalar_context


# feature_player: [ 2 95  0 12 15  0 12  0  0  0  0]
# player_id, minerals, vespene, food_used, food_cap, food_army, food_workers, idle_worker_count, army_count, warp_gate_count, larva_count 
def get_agent_statistics(score_by_category):
  score_by_category = score_by_category.flatten() / 1000.0
  agent_statistics = np.log(score_by_category + 1)

  return agent_statistics


race_list = ["Protoss", "Terran", "Zerg", "Unknown"]
def get_race_onehot(home_race, away_race):
  home_race_index = race_list.index(home_race)
  away_race_index = race_list.index(away_race)
  home_race_onehot = np.identity(5)[home_race_index:home_race_index+1]
  away_race_onehot = np.identity(5)[away_race_index:away_race_index+1]

  return np.array([home_race_onehot[0], away_race_onehot[0]]).flatten()
