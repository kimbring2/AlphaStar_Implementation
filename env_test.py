from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
import sys

import random
import time

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_name = 'Simple64'
players = [sc2_env.Agent(sc2_env.Race['terran']), 
           sc2_env.Bot(sc2_env.Race['protoss'], sc2_env.Difficulty.very_easy)]
feature_screen_size = 84
feature_minimap_size = 64
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
step_mul = 8
game_steps_per_episode = None
disable_fog = False
visualize = True

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

_SELECT_POINT = actions.FUNCTIONS.select_point.id


obs = env.reset()
for i in range(0, 1000):
  #print("i: " + str(i))
  time.sleep(0.5)

  target_0 = [random.randint(0,25),random.randint(0,25)]
  target_1 = [random.randint(0,25),random.randint(0,25)]

  action_0 = [actions.FUNCTIONS.no_op()]
  action_1 = [actions.FUNCTIONS.Move_screen("now", target_0)]
  action_2 = [actions.FUNCTIONS.select_point("select", target_0)]
  action_3 = [actions.FUNCTIONS.select_rect("select", target_0, target_1)]

  obs_feature_units = obs[0][3]['feature_units']
  obs_feature_player = obs[0][3]['player']

  self_CommandCenter = [unit for unit in obs_feature_units if unit.unit_type == units.Terran.CommandCenter and unit.alliance == _PLAYER_SELF]
  self_SCV = [unit for unit in obs_feature_units if unit.unit_type == units.Terran.SCV and unit.alliance == _PLAYER_SELF]
  
  print("obs_feature_player.minerals :" + str(obs_feature_player.minerals))
  #print("self_CommandCenter[0].y :" + str(self_CommandCenter[0].y))
  print("")
  print("")
  print("")

  #print("obs_feature_units: " + str(obs_feature_units))
  #print("obs[2]: " + str(obs[2]))
  #roaches = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_ENEMY]

  #print("marines: " + str(marines))
  #print("marines[0].x: " + str(marines[0].x))
  #print("marines[0].y: " + str(marines[0].y))
  #print("")
  target = [self_SCV[0].x, self_SCV[0].y]
  action = [actions.FUNCTIONS.select_point("select", target)]
  #action = [actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", self_scv[0], pylon_xy)]
  obs = env.step(action)
  #for marine in marines:
    #print("len(marines): " + str(len(marines)))
    #print("marine.x: " + str(marine.x))
    #print("marine.y: " + str(marine.y))
    #print("")

