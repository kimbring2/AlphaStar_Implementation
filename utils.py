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
_MINIMAP_CAMERA = features.MINIMAP_FEATURES.camera.index
_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index

all_unit_list = [0, 37, 45, 48, 317, 21, 341, 342, 18, 27, 132, 20, 5, 47, 21, 
                 19, 483, 51, 28, 42, 53, 268, 472, 49, 41, 830, 105, 9, 1680, 110]

# Marine = 48
# Zergling = 105
# Baneling = 9
# Roach = 110
# Mineral = 1680
# Beacon = 317

#essential_unit_list = [0, 45, 48, 317, 21, 341, 18, 27, 20, 19, 483, 500] # For Simple64
#essential_unit_list = [0, 48, 105, 9]  # For Minigame
essential_unit_list = [0, 48, 1680]


def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_UNIT_TYPE:
      scale = len(essential_unit_list)
      layer = np.zeros([scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(len(all_unit_list)):
        indy, indx = (screen[i] == all_unit_list[j]).nonzero()

        if all_unit_list[j] in essential_unit_list:
          unit_index = essential_unit_list.index(all_unit_list[j])
          layer[unit_index, indy, indx] = 1
        else:
          layer[-1, indy, indx] = 1

      layers.append(layer)
    elif i == _SCREEN_SELECTED:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1

      layers.append(layer)
    elif i == _SCREEN_UNIT_HIT_POINTS:
      layers.append(np.log(screen[i:i+1] + 1) / np.log(features.SCREEN_FEATURES[i].scale))

  return np.concatenate(layers, axis=0)


def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif i == _MINIMAP_CAMERA or i == _MINIMAP_PLAYER_RELATIVE:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
        
      layers.append(layer)  

  return np.concatenate(layers, axis=0)


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])
FLAT_FEATURES = [
  FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
  FlatFeature(1,  features.FeatureType.SCALAR, 10000, 'minerals'),
  FlatFeature(2,  features.FeatureType.SCALAR, 10000, 'vespene'),
  FlatFeature(3,  features.FeatureType.SCALAR, 200, 'food_used'),
  FlatFeature(4,  features.FeatureType.SCALAR, 200, 'food_cap'),
  FlatFeature(5,  features.FeatureType.SCALAR, 200, 'food_army'),
  FlatFeature(6,  features.FeatureType.SCALAR, 200, 'food_workers'),
  FlatFeature(7,  features.FeatureType.SCALAR, 200, 'idle_worker_count'),
  FlatFeature(8,  features.FeatureType.SCALAR, 200, 'army_count'),
  FlatFeature(9,  features.FeatureType.SCALAR, 200, 'warp_gate_count'),
  FlatFeature(10, features.FeatureType.SCALAR, 200, 'larva_count'),
]
def preprocess_player(player):
  layers = []
  for s in FLAT_FEATURES:
    if s.index == 1 or s.index == 2:
      out = np.log(player[s.index] + 1) / np.log(s.scale)
      layers.append(out)
    else:
      out = player[s.index] / s.scale
      layers.append(out)

  return np.array(layers)


def preprocess_available_actions(available_action):
    available_actions = np.zeros(_NUM_FUNCTIONS, dtype=np.float64)
    available_actions[available_action] = 1

    return available_actions


def preprocess_feature_units(feature_units, feature_screen_size):
    feature_units_list = []
    feature_units_length = len(feature_units)
    for i, feature_unit in enumerate(feature_units):
      #if feature_unit.unit_type == 19:
      #  print("feature_unit: ", feature_unit)

      feature_unit_length = len(feature_unit) 

      feature_unit_list = []
      feature_unit_list.append(feature_unit.unit_type / 2000)
      feature_unit_list.append(feature_unit.alliance / 4)
      feature_unit_list.append(feature_unit.health / 10000)
      feature_unit_list.append(feature_unit.shield / 10000)
      feature_unit_list.append(feature_unit.x / 100)
      feature_unit_list.append(feature_unit.y / 100) 
      feature_unit_list.append(feature_unit.is_selected)
      feature_unit_list.append(feature_unit.build_progress / 500)

      #print("feature_unit.x / (feature_screen_size + 1): ", feature_unit.x / (feature_screen_size + 1))
      #print("feature_unit.y / (feature_screen_size + 1): ", feature_unit.y / (feature_screen_size + 1))

      feature_units_list.append(feature_unit_list)
      #print("i: ", i)
      if i >= 49:
        break

    if feature_units_length < 50:
      for i in range(feature_units_length, 50):
        feature_units_list.append(np.zeros(8))

    entity_array = np.array(feature_units_list)
    
    return entity_array


SingleSelectFeature = namedtuple('SingleSelectFeature', ['index', 'type', 'scale', 'name'])
SINGLE_SELECT_FEATURES = [
  SingleSelectFeature(0,  features.FeatureType.SCALAR, len(essential_unit_list), 'unit_type'),
  SingleSelectFeature(1,  features.FeatureType.SCALAR, 4, 'player_relative'),
  SingleSelectFeature(2,  features.FeatureType.SCALAR, 2000, 'health'),
]
def preprocess_single_select(single_select):
  if len(single_select) != 0:
    single_select = single_select[0]
    layers = []
    for s in SINGLE_SELECT_FEATURES:
      if s.index == 2:
        out = np.log(single_select[s.index] + 1) / np.log(s.scale)
        layers.append(out)
      elif s.index == 0:
        out = essential_unit_list.index(single_select[s.index]) / s.scale
        layers.append(out)
      else:
        out = single_select[s.index] / s.scale
        layers.append(out)
  else:
    layers = [0.0, 0.0, 0.0]

  return np.array(layers)


ScoreCumulativeFeature = namedtuple('ScoreCumulativeFeature', ['index', 'type', 'scale', 'name'])
SCORE_CUMULATIVE_FEATURES = [
  ScoreCumulativeFeature(0,  features.FeatureType.SCALAR, 25000, ' score '),
  ScoreCumulativeFeature(1,  features.FeatureType.SCALAR, 5000, 'idle_production_time'),
  ScoreCumulativeFeature(2,  features.FeatureType.SCALAR, 10000, 'idle_worker_time'),
  ScoreCumulativeFeature(3,  features.FeatureType.SCALAR, 10000, 'total_value_units'),
  ScoreCumulativeFeature(4,  features.FeatureType.SCALAR, 10000, 'total_value_structures'),
  ScoreCumulativeFeature(5,  features.FeatureType.SCALAR, 10000, 'killed_value_units'),
  ScoreCumulativeFeature(6,  features.FeatureType.SCALAR, 10000, 'killed_value_structures'),
  ScoreCumulativeFeature(7,  features.FeatureType.SCALAR, 10000, 'collected_minerals'),
  ScoreCumulativeFeature(8,  features.FeatureType.SCALAR, 10000, 'collected_vespene'),
  ScoreCumulativeFeature(9,  features.FeatureType.SCALAR, 2000, 'collection_rate_minerals'),
  ScoreCumulativeFeature(10, features.FeatureType.SCALAR, 2000, 'collection_rate_vespene'),
  ScoreCumulativeFeature(11, features.FeatureType.SCALAR, 10000, 'spent_minerals'),
  ScoreCumulativeFeature(12, features.FeatureType.SCALAR, 10000, 'spent_vespene'),
]
def preprocess_score_cumulative(score_cumulative):
  layers = []
  for s in SCORE_CUMULATIVE_FEATURES:
    if s.index == 9 or s.index == 10:
      out = score_cumulative[s.index] / s.scale
      layers.append(out)
    else:
      out = np.log(score_cumulative[s.index] + 1) / np.log(s.scale)
      out = score_cumulative[s.index] / s.scale
      layers.append(out)

  return np.array(layers)


def preprocess_build_queue(build_queue):
  build_queue_length = len(build_queue)
  if build_queue_length > 5:
    build_queue_length = 5

  layers = [0.0, 0.0, 0.0, 0.0, 0.0]
  for i in range(0, build_queue_length):
    layers[i] = (essential_unit_list.index(build_queue[i][0]) / len(essential_unit_list))

  return np.array(layers)


def preprocess_multi_select(multi_select):
  multi_select_length = len(multi_select)
  if multi_select_length > 10:
    multi_select_length = 10

  layers = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  for i in range(0, multi_select_length):
    layers[i] = (essential_unit_list.index(multi_select[i][0]) / len(essential_unit_list))

  return np.array(layers)


def positional_encoding(max_position, embedding_size, add_batch_dim=False):
    positions = np.arange(max_position)
    angle_rates = 1 / np.power(10000, (2 * (np.arange(embedding_size)//2)) / np.float32(embedding_size))
    angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    if add_batch_dim:
        angle_rads = angle_rads[np.newaxis, ...]

    return tf.cast(angle_rads, dtype=tf.float32)