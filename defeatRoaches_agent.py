# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import cv2
import math
import statistics

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import random
import time

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))


class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not beacon:
        return FUNCTIONS.no_op()
      beacon_center = numpy.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", beacon_center)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class CollectMineralShardsFeatureUnits(base_agent.BaseAgent):
  """An agent for solving the CollectMineralShards map with feature units.
  Controls the two marines independently:
  - select marine
  - move to nearest mineral shard that wasn't the previous target
  - swap marine and repeat
  """

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsFeatureUnits, self).setup(obs_spec, action_spec)
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(CollectMineralShardsFeatureUnits, self).reset()
    self._marine_selected = False
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CollectMineralShardsFeatureUnits, self).step(obs)
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    if not marines:
      return FUNCTIONS.no_op()
    marine_unit = next((m for m in marines
                        if m.is_selected == self._marine_selected), marines[0])
    marine_xy = [marine_unit.x, marine_unit.y]

    if not marine_unit.is_selected:
      # Nothing selected or the wrong marine is selected.
      self._marine_selected = True
      return FUNCTIONS.select_point("select", marine_xy)

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      # Find and move to the nearest mineral.
      minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == _PLAYER_NEUTRAL]

      if self._previous_mineral_xy in minerals:
        # Don't go for the same mineral shard as other marine.
        minerals.remove(self._previous_mineral_xy)

      if minerals:
        # Find the closest.
        distances = numpy.linalg.norm(
            numpy.array(minerals) - numpy.array(marine_xy), axis=1)
        closest_mineral_xy = minerals[numpy.argmin(distances)]

        # Swap to the other marine.
        self._marine_selected = False
        self._previous_mineral_xy = closest_mineral_xy
        return FUNCTIONS.Move_screen("now", closest_mineral_xy)

    return FUNCTIONS.no_op()


def plot_graphs_tuple_np(graphs_tuple):
  networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
  num_graphs = len(networkx_graphs)
  _, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  for graph, ax in zip(networkx_graphs, axes):
    plot_graph_networkx(graph, ax)


def plot_graphs_tuple_np(graphs_tuple):
  networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
  num_graphs = len(networkx_graphs)
  _, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  for graph, ax in zip(networkx_graphs, axes):
    plot_graph_networkx(graph, ax)


def plot_graph_networkx(graph, ax, pos=None):
  node_labels = {node: "{:.3g}".format(data["features"][0])
                 for node, data in graph.nodes(data=True)
                 if data["features"] is not None}
  edge_labels = {(sender, receiver): "{:.3g}".format(data["features"][0])
                 for sender, receiver, data in graph.edges(data=True)
                 if data["features"] is not None}
  global_label = ("{:.3g}".format(graph.graph["features"][0])
                  if graph.graph["features"] is not None else None)

  if pos is None:
    pos = nx.spring_layout(graph)
  nx.draw_networkx(graph, pos, ax=ax, labels=node_labels)

  if edge_labels:
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

  if global_label:
    plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  return pos

'''
python -m pysc2.bin.agent --map DefeatRoaches --agent relation_agent.DefeatRoaches --use_feature_units True
'''
class DefeatRoaches(base_agent.BaseAgent):
  """An agent specifically for solving the DefeatRoaches map."""
  #print("FUNCTIONS: " + str(FUNCTIONS))
  '''
   0/no_op                                              ()
   1/move_camera                                        (1/minimap [64, 64])
   2/select_point                                       (6/select_point_act [4]; 0/screen [84, 84])
   3/select_rect                                        (7/select_add [2]; 0/screen [84, 84]; 2/screen2 [84, 84])
   4/select_control_group                               (4/control_group_act [5]; 5/control_group_id [10])
   5/select_unit                                        (8/select_unit_act [4]; 9/select_unit_id [500])
 453/Stop_quick                                         (3/queued [2])
   7/select_army                                        (7/select_add [2])
 451/Smart_screen                                       (3/queued [2]; 0/screen [84, 84])
 452/Smart_minimap                                      (3/queued [2]; 1/minimap [64, 64])
 331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])
 332/Move_minimap                                       (3/queued [2]; 1/minimap [64, 64])
 333/Patrol_screen                                      (3/queued [2]; 0/screen [84, 84])
 334/Patrol_minimap                                     (3/queued [2]; 1/minimap [64, 64])
  12/Attack_screen                                      (3/queued [2]; 0/screen [84, 84])
  13/Attack_minimap                                     (3/queued [2]; 1/minimap [64, 64])
 274/HoldPosition_quick                                 (3/queued [2])
  '''
  def reset(self):
    super(DefeatRoaches, self).reset()
    self._flag_1 = False
    self._flag_2 = False
    self._flag_3 = False
    self._flag_4 = False

    self._flag_5 = True
    self._flag_6 = False
    self._flag_7 = False

    self.step_num = 1

    self.screen_direction = None
    self.flag_temp_1 = 0

  def get_dis(self, pos1, pos2):
    return math.sqrt( ((pos1[0]-pos2[0])**2)+((pos1[1]-pos2[1])**2) )

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    self.step_num = self.step_num + 1

    if (self._flag_5 == True):
      marines = [unit for unit in obs.observation.feature_units
                 if unit.alliance == _PLAYER_SELF]
      if not marines:
        return FUNCTIONS.no_op()

      marine_x_list = []
      marine_y_list = []
      for marine in marines:
        marine_x_list.append(marine.x)
        marine_y_list.append(marine.y)

      if (len(marines) > 1):
        remain_marines_x_list = []
        remain_marines_y_list = []
        for i in range(0, len(marines)):
          remain_marines_x_list.append(marines[i].x)
          remain_marines_y_list.append(marines[i].y)

        remain_marines_x_mean = min(remain_marines_x_list)
        remain_marines_y_mean = min(remain_marines_y_list)

      remain_marines_x_list.sort()
      remain_marines_y_list.sort()

      if (self._flag_1 == False):
        self._flag_1 = True
        return FUNCTIONS.select_point("select", [remain_marines_x_list[0], remain_marines_y_list[0]]) 

      if (self._flag_3 == False):
          self._flag_3 = True
          return FUNCTIONS.select_control_group("set", 0)

      if (self._flag_2 == False):
        self._flag_2 = True
        return FUNCTIONS.select_rect("select", 
                                     [remain_marines_x_list[1], remain_marines_y_list[1]], 
                                     [remain_marines_x_list[-1], remain_marines_y_list[-1]])
      if (self._flag_4 == False):
          self._flag_4 = True
          return FUNCTIONS.select_control_group("set", 1)

      self._flag_1 = False
      self._flag_2 = False
      self._flag_3 = False
      self._flag_4 = False

      self._flag_5 = False
      self._flag_6 = True
      self._flag_7 = False

      return FUNCTIONS.no_op()
    elif (self._flag_6 == True):
      if (self._flag_1 == False):
        self._flag_1 = True
        return FUNCTIONS.select_control_group("recall", 0) 

      if (self._flag_2 == False):
        self._flag_2 = True

        selected_marines = [unit for unit in obs.observation.feature_units
                          if unit.is_selected == 1]
        if ( (not selected_marines) | (len(selected_marines) != 1) ):
          return FUNCTIONS.no_op()

        selected_marines_x_list = []
        selected_marines_y_list = []
        for selected_marine in selected_marines:
          selected_marines_x_list.append(selected_marine.x)
          selected_marines_y_list.append(selected_marine.y)

        selected_marines_x_mean = statistics.mean(selected_marines_x_list)
        selected_marines_y_mean = statistics.mean(selected_marines_y_list)

        roaches = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]
        if not roaches:
          return FUNCTIONS.no_op()

        roache_x_list = []
        roache_y_list = []
        for roache in roaches:
          roache_x_list.append(roache.x)
          roache_y_list.append(roache.y)

        roache_x_mean = statistics.mean(roache_x_list)
        roache_y_mean = statistics.mean(roache_y_list)

        if ( (selected_marines_x_mean >= roache_x_mean) & (selected_marines_y_mean >= roache_y_mean) ):
          x_point = roache_x_mean + 25
          y_point = roache_y_mean + 20
        elif ( (selected_marines_x_mean >= roache_x_mean) & (selected_marines_y_mean < roache_y_mean) ):
          x_point = roache_x_mean + 25
          y_point = roache_y_mean + 20
        elif ( (selected_marines_x_mean < roache_x_mean) & (selected_marines_y_mean >= roache_y_mean) ):
          x_point = roache_x_mean - 25
          y_point = roache_y_mean + 20
        elif ( (selected_marines_x_mean < roache_x_mean) & (selected_marines_y_mean < roache_y_mean) ):
          x_point = roache_x_mean - 25
          y_point = roache_y_mean + 20

        if ( (x_point > 83) | (x_point < 1) | (y_point > 83) | (y_point < 1) ):
          return FUNCTIONS.no_op()
          
        return FUNCTIONS.Move_screen("now", [x_point, y_point])
      if (self._flag_3 == False):
        self._flag_3 = True
        return FUNCTIONS.select_control_group("recall", 1) 

      if (self._flag_4 == False):
        self._flag_4 = True

        selected_marines = [unit for unit in obs.observation.feature_units
                          if unit.is_selected == 1]

        if ( (not selected_marines) | (len(selected_marines) == 1) ):
          return FUNCTIONS.no_op()

        selected_marines_x_list = []
        selected_marines_y_list = []
        for selected_marine in selected_marines:
          selected_marines_x_list.append(selected_marine.x)
          selected_marines_y_list.append(selected_marine.y)

        selected_marines_x_mean = statistics.mean(selected_marines_x_list)
        selected_marines_y_mean = statistics.mean(selected_marines_y_list)

        roaches = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]                 
        if not roaches:
          return FUNCTIONS.no_op()

        roache_x_list = []
        roache_y_list = []
        for roache in roaches:
          roache_x_list.append(roache.x)
          roache_y_list.append(roache.y)

        roache_x_mean = statistics.mean(roache_x_list)
        roache_y_mean = statistics.mean(roache_y_list)

        if ( (selected_marines_x_mean >= roache_x_mean) & (selected_marines_y_mean >= roache_y_mean) ):
          x_point = roache_x_mean + 25
          y_point = roache_y_mean - 20
        elif ( (selected_marines_x_mean >= roache_x_mean) & (selected_marines_y_mean < roache_y_mean) ):
          x_point = roache_x_mean + 25
          y_point = roache_y_mean - 20
        elif ( (selected_marines_x_mean < roache_x_mean) & (selected_marines_y_mean >= roache_y_mean) ):
          x_point = roache_x_mean - 25
          y_point = roache_y_mean - 20
        elif ( (selected_marines_x_mean < roache_x_mean) & (selected_marines_y_mean < roache_y_mean) ):
          x_point = roache_x_mean - 25
          y_point = roache_y_mean - 20

        if ( (x_point > 83) | (x_point < 1) | (y_point > 83) | (y_point < 1) ):
          return FUNCTIONS.no_op()

        return FUNCTIONS.Move_screen("now", [x_point, y_point])

      self._flag_1 = False
      self._flag_2 = False
      self._flag_3 = False
      self._flag_4 = False

      self._flag_5 = False
      self._flag_6 = False
      self._flag_7 = True
    elif (self._flag_7 == True):
      #selected_marines = [unit for unit in obs.observation.feature_units
      #                    if unit.is_selected == 1]
      #print("selected_marines[0]: " + str(selected_marines[0]))

      if (self._flag_1 == False):
        self._flag_1 = True
        return FUNCTIONS.select_control_group("recall", 0)

      if (self._flag_2 == False):
        selected_marines = [unit for unit in obs.observation.feature_units
                            if unit.is_selected == 1]
        if ( (not selected_marines) | (len(selected_marines) != 1) | selected_marines[0][26] == 1):
          return FUNCTIONS.no_op()

        self._flag_2 = True
        marine_position = [selected_marines[0].x, selected_marines[0].y] 

        roache_position_list = []
        roaches = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]
        for roache in roaches:
            roache_position_list.append([roache.x, roache.y])
        if not roaches:
          return FUNCTIONS.no_op()
        distance_list = []

        for roache_position in roache_position_list:
            dis = self.get_dis(marine_position, roache_position)
            distance_list.append(dis)

        min_dis_index = distance_list.index(min(distance_list))
        min_dis_roache_pos = roache_position_list[min_dis_index]

        if ( (marine_position[0] >= min_dis_roache_pos[0]) & (marine_position[1] >= min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] + 10
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] >= min_dis_roache_pos[0]) & (marine_position[1] < min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] + 10
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] < min_dis_roache_pos[0]) & (marine_position[1] >= min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] - 10
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] < min_dis_roache_pos[0]) & (marine_position[1] < min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] - 10
            y_point = min_dis_roache_pos[1] + 15

        if ( (x_point > 83) | (x_point < 1) | (y_point > 83) | (y_point < 1) ):
          return FUNCTIONS.no_op()
        return FUNCTIONS.Move_screen("now", [x_point, y_point])

      if (self._flag_3 == False):
        selected_marines = [unit for unit in obs.observation.feature_units
                            if unit.is_selected == 1]
        if ( (not selected_marines) | (len(selected_marines) != 1) | selected_marines[0][26] == 1):
          return FUNCTIONS.no_op()

        self._flag_3 = True
        marine_position = [selected_marines[0].x, selected_marines[0].y] 

        roache_position_list = []
        roaches = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]
        for roache in roaches:
            roache_position_list.append([roache.x, roache.y])
        if not roaches:
          return FUNCTIONS.no_op()
        distance_list = []

        for roache_position in roache_position_list:
            dis = self.get_dis(marine_position, roache_position)
            distance_list.append(dis)

        min_dis_index = distance_list.index(min(distance_list))
        min_dis_roache_pos = roache_position_list[min_dis_index]

        if ( (marine_position[0] >= min_dis_roache_pos[0]) & (marine_position[1] >= min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] + 35
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] >= min_dis_roache_pos[0]) & (marine_position[1] < min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] + 35
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] < min_dis_roache_pos[0]) & (marine_position[1] >= min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] - 35
            y_point = min_dis_roache_pos[1] + 15
        elif ( (marine_position[0] < min_dis_roache_pos[0]) & (marine_position[1] < min_dis_roache_pos[1]) ):
            x_point = min_dis_roache_pos[0] - 35
            y_point = min_dis_roache_pos[1] + 15

        if ( (x_point > 83) | (x_point < 1) | (y_point > 83) | (y_point < 1) ):
          return FUNCTIONS.no_op()
        return FUNCTIONS.Move_screen("now", [x_point, y_point])
      return FUNCTIONS.no_op()

      self._flag_1 = False
      self._flag_2 = False
      self._flag_3 = False
      
    return FUNCTIONS.no_op()
    '''
      if ( (marine_unit.x > roache_x_mean) & (marine_unit.y > roache_y_mean) ):
        x_point = roache_x_mean + 25
        y_point = roache_y_mean + 25
      elif ( (marine_unit.x > roache_x_mean) & (marine_unit.y < roache_y_mean) ):
        x_point = roache_x_mean + 25
        y_point = roache_y_mean - 25
      elif ( (marine_unit.x < roache_x_mean) & (marine_unit.y > roache_y_mean) ):
        x_point = roache_x_mean - 25
        y_point = roache_y_mean + 25
      elif ( (marine_unit.x < roache_x_mean) & (marine_unit.y < roache_y_mean) ):
        x_point = roache_x_mean - 25
        y_point = roache_y_mean - 25
      else:
        x_point = marine_unit.x
        y_point = marine_unit.y

      if ( (x_point > 83) | (x_point < 1) | (y_point > 83) | (y_point < 1) ):
          return FUNCTIONS.no_op()
      else:
        return FUNCTIONS.Move_screen("now", [x_point, y_point])
    '''
    '''
    #marine_unit = next((m for m in marines
    #                    if m.is_selected == self._marine_selected), marines[0])
    #marine_xy = [marine_unit.x, marine_unit.y]
    marine_num = 0
    marine_list = []
    for m in marines:
      marine_num = marine_num + 1
      if (marine_num <= 9):
        marine_list.append([m.x, m.y, 0])
    #print("marine_list: " + str(marine_list))

    roaches = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_ENEMY]
    if not roaches:
      return FUNCTIONS.no_op()

    roache_list = []
    for r in roaches:
      roache_list.append([r.x, r.y, 1])
    #print("roache_list: " + str(roache_list))

    nodes = (np.array(marine_list + roache_list)).astype(np.float32)
    #print("nodes " + str(nodes))
    #print("nodes[:,2] " + str(nodes[:,2]))

    marines_index = np.where(nodes[:,2] == 0)
    marines_index = marines_index[0]
    roaches_index = np.where(nodes[:,2] == 1)
    roaches_index = roaches_index[0]
    #print("marines_index: " + str(marines_index))
    #print("roaches_index: " + str(roaches_index))

    edges = []
    senders = []
    receivers = []
    for marine_index in marines_index.tolist():
      for roache_index in roaches_index.tolist():
        dis = self.get_dis((np.array(nodes))[marine_index][0:2], (np.array(nodes))[roache_index][0:2])
        #print("dis: " + str(dis))
        edges.append([dis, 0])
        senders.append(marine_index)
        receivers.append(roache_index)

    edges = (np.array(edges)).astype(np.float32)
    
    for roache_index in roaches_index.tolist():
      for marine_index in marines_index.tolist():
        dis = self.get_dis((np.array(nodes))[roache_index][0:2], (np.array(nodes))[marine_index][0:2])
        #dis = self.get_dis(marine, roache)
        #print("dis: " + str(dis))
        edges.append([dis, 1])
        senders.append(roache_index)
        receivers.append(marine_index)
    
    for marine_index in marines_index.tolist():
      for marine_index in marines_index.tolist():
        #dis = self.get_dis(marine, roache)
        #print("dis: " + str(dis))
        edges.append(1)
        senders.append(marine_index)
        receivers.append(marine_index)

    for roache_index in roaches_index.tolist():
      for roache_index in roaches_index.tolist():
        #dis = self.get_dis(marine, roache)
        #print("dis: " + str(dis))
        edges.append(1)
        senders.append(roache_index)
        receivers.append(roache_index)
    
    data_dict = {
      "nodes": nodes,
      "edges": edges,
      "senders": senders,
      "receivers": receivers,
      "globals": None
    }

    tf.reset_default_graph()
    OUTPUT_EDGE_SIZE = 10
    OUTPUT_NODE_SIZE = 11
    OUTPUT_GLOBAL_SIZE = 12
    encoder_network = modules.GraphIndependent(
        node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
        edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
        global_model_fn=None,
    )

    core_network = modules.RelationNetwork(
        edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
        global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE)
    )

    decoder_network = modules.GraphIndependent(
        node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
        edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
        global_model_fn=None,
    )

    input_graphs = utils_tf.data_dicts_to_graphs_tuple([data_dict])
    runnable_in_session_graph = utils_tf.make_runnable_in_session(input_graphs)

    latent_graphs = encoder_network(runnable_in_session_graph)
    latent_graphs = core_network(latent_graphs)
    decoded_op = decoder_network(runnable_in_session_graph)

    print("Output edges size: {}".format(decoded_op.edges.shape[-1]))  # Equal to OUTPUT_EDGE_SIZE
    print("Output nodes size: {}".format(decoded_op.nodes.shape[-1]))  # Equal to OUTPUT_NODE_SIZE
    #print("Output globals size: {}".format(decoded_op.globals.shape[-1]))  # Equal to OUTPUT_GLOBAL_SIZE
    #plot_graphs_tuple_np(graphs_tuple_np)
    #plt.show()

    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative

      roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
      #print("roaches: " + str(roaches))
      if not roaches:
        return FUNCTIONS.no_op()

      # Find the roach with max y coord.
      target = roaches[numpy.argmax(numpy.array(roaches)[:, 1])]
      #print("target: " + str(target))
      #return FUNCTIONS.Attack_screen("now", [0, 0])
      #return FUNCTIONS.Stop_quick("now")
      return FUNCTIONS.select_unit("select", 48)
    '''
    
    #if FUNCTIONS.select_army.id in obs.observation.available_actions:
    #  return FUNCTIONS.select_army("select")

    #return FUNCTIONS.no_op()