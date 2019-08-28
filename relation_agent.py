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

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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
    self._marine_selected = False

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)

    #print("obs.observation: " + str(obs.observation))
    #print("")

    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    #print("marines: " + str(marines))
    if not marines:
      return FUNCTIONS.no_op()

    marine_unit = next((m for m in marines
                        if m.is_selected == self._marine_selected), marines[0])
    marine_xy = [marine_unit.x, marine_unit.y]
    
    #print("marine_xy: " + str(marine_xy))
    #print("marine_xy[0]: " + str(marine_xy[0]))
    #print("")
    if not marine_unit.is_selected:
      # Nothing selected or the wrong marine is selected.
      self._marine_selected = True
      return FUNCTIONS.select_point("select", marine_xy)

    if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
      return FUNCTIONS.Attack_screen("now", [0, 0])

    '''
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
