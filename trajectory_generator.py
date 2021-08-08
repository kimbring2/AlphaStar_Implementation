#!/usr/bin/env python
from pysc2.lib import features, point, actions, units
from pysc2.env.environment import TimeStep, StepType
from pysc2.env import sc2_env, available_actions_printer
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

import importlib
import random
import sys
import glob
import argparse
import hickle as hkl 
import numpy as np

from absl import app, flags
FLAGS = flags.FLAGS
FLAGS(['trajectory.py'])

parser = argparse.ArgumentParser(description='Trajetory File Generation')
parser.add_argument('--replay_path', type=str, help='Path of replay file')
parser.add_argument('--player_1', type=str, default='Terran', help='Race of player 1')
parser.add_argument('--player_2', type=str, default='Terran', help='Race of player 2')
parser.add_argument('--mmr', type=int, default=2500, help='Threshold of mmr score ')
parser.add_argument('--saving_path', type=str, help='Path for saving proprocessed replay file')

arguments = parser.parse_args()

class Trajectory(object):
	def __init__(self, source, home_race_name, away_race_name, replay_filter, filter_repeated_camera_moves=True):
	    self.source = source
	    self.home_race_name = home_race_name
	    self.away_race_name = away_race_name
	    self.replay_filter = replay_filter
	    self.filter_repeated_camera_moves = filter_repeated_camera_moves

	    self.home_BO = None

	    self.home_observation = []
	    self.home_feature_screen = []
	    self.home_feature_minimap = []
	    self.home_player = []
	    self.home_feature_units = []
	    self.home_game_loop = []
	    self.home_available_actions = []

	    self.home_action = []

	    self.away_BU = None
	    self.away_trajectory = []

	    self.home_score_cumulative = None
	    self.away_score_cumulative = None

	def get_BO(self, player):
		if player == 0:
			return self.home_BO
		else:
			return self.away_BU

	def generate_trajectory(self):
		function_dict = {}
		for _FUNCTION in actions._FUNCTIONS:
		    function_dict[_FUNCTION.ability_id] = _FUNCTION.name

		race_list = ['Terran', 'Zerg', 'Protoss']

		"""How many agent steps the agent has been trained for."""
		run_config = run_configs.get()
		sc2_proc = run_config.start()
		controller = sc2_proc.controller

		root_path =  self.source
		replay_file_path_list = glob.glob(root_path + '*.*')
		for replay_file_path in replay_file_path_list:
			#print("replay_file_path: " + str(replay_file_path))

			#replay_file_path = random.choice(file_list)
			replay_file_name = replay_file_path.split('/')[-1].split('.')[0]
			#print("replay_file_name: {}".format(replay_file_name))
			#print("replay_file_path: {}".format(replay_file_path))
			try: 
				replay_data = run_config.replay_data(replay_file_path)
				ping = controller.ping()
				info = controller.replay_info(replay_data)

				player0_race = info.player_info[0].player_info.race_actual
				player0_mmr = info.player_info[0].player_mmr
				player0_apm = info.player_info[0].player_apm
				player0_result = info.player_info[0].player_result.result
				
				home_race = race_list.index(self.home_race_name) + 1
				if (home_race == player0_race):
					print("player0_race pass")
					#continue
				else:
					print("player0_race fail")
					#continue

				if (player0_mmr >= self.replay_filter):
					print("player0_mmr pass")
					#continue
				else:
					print("player0_mmr fail")
					#continue
				
				player1_race = info.player_info[0].player_info.race_actual
				player1_mmr = info.player_info[0].player_mmr
				player1_apm = info.player_info[0].player_apm
				player1_result = info.player_info[0].player_result.result

				away_race = race_list.index(self.away_race_name) + 1
				if (away_race == player1_race):
					print("player1_race pass ")
					#continue
				else:
					print("player1_race fail ")
					#continue

				if (player1_mmr >= self.replay_filter):
					print("player1_mmr pass ")
					#continue
				else:
					print("player1_mmr fail")
					#continue
				
				screen_size_px = (32, 32)
				minimap_size_px = (32, 32)
				player_id = 1
				discount = 1.
				step_mul = 2

				screen_size_px = point.Point(*screen_size_px)
				minimap_size_px = point.Point(*minimap_size_px)
				interface = sc_pb.InterfaceOptions(raw=True, score=True,
					feature_layer=sc_pb.SpatialCameraSetup(width=24))
				screen_size_px.assign_to(interface.feature_layer.resolution)
				minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

				map_data = None
				if info.local_map_path:
					map_data = run_config.map_data(info.local_map_path)

				_episode_length = info.game_duration_loops
				_episode_steps = 0

				controller.start_replay(sc_pb.RequestStartReplay(replay_data=replay_data, 
					map_data=map_data, options=interface,
					observed_player_id=player_id))

				_state = StepType.FIRST

				if (info.HasField("error") or
				                    info.base_build != ping.base_build or  # different game version
				                    info.game_duration_loops < 1000 or
				                    len(info.player_info) != 2):
					# Probably corrupt, or just not interesting.
					print("error")
					continue

				feature_screen_size = 32
				feature_minimap_size = 32
				rgb_screen_size = None
				rgb_minimap_size = None
				action_space = None
				use_feature_units = True
				aif = sc2_env.parse_agent_interface_format(
					feature_screen=feature_screen_size,
					feature_minimap=feature_minimap_size,
					rgb_screen=rgb_screen_size,
					rgb_minimap=rgb_minimap_size,
					action_space=action_space,
					use_feature_units=use_feature_units)

				_features = features.features_from_game_info(controller.game_info(), agent_interface_format=aif)

				build_info = []
				build_name = []
				replay_step = 0

				#print("_episode_length: {}".format(_episode_length))
				for replay_step in range(0, _episode_length):
					controller.step(step_mul)
					obs = controller.observe()
					if obs.player_result: # Episode over.
						_state = StepType.LAST
						discount = 0
					else:
						discount = discount
						_episode_steps += step_mul

					agent_obs = _features.transform_obs(obs)
					if len(obs.actions) != 0:
						exec_actions = []
						for ac in obs.actions:
							exec_act = _features.reverse_action(ac)

							a_0 = int(exec_act.function)
							a_l = []
							#print("exec_act.arguments: {}".format(exec_act.arguments))
							for argument in exec_act.arguments:
								#print("argument: {}".format(argument))
								if str(type(argument[0])) != "<class 'int'>": 
									a_l.append(argument[0].value)
								else:
									a_l.append(argument)
							
							exec_actions.append([a_0, a_l])

							#print("a_0: {}".format(a_0))
							#print("a_l: {}".format(a_l))
						self.home_action.append(exec_actions)
					else:
						exec_actions = []
						a_0 = 0
						a_l = [0]
						exec_actions.append([a_0, a_l])
						if replay_step % 16 == 0 or _state == StepType.LAST:
							self.home_action.append(exec_actions)
							pass
						else:
							continue
					
					#print("")

					done = 0
					if _state == StepType.LAST:
						done = 1
					
					self.home_feature_screen.append(agent_obs['feature_screen'])
					self.home_feature_minimap.append(agent_obs['feature_minimap'])
					self.home_player.append(agent_obs['player'])
					self.home_feature_units.append(agent_obs['feature_units'])
					self.home_game_loop.append(agent_obs['game_loop'])
					self.home_available_actions.append(agent_obs['available_actions'])

					step = TimeStep(step_type=_state, reward=0,
				                       discount=discount, observation=agent_obs)

					if _state == StepType.LAST:
						file_path = arguments.saving_path + replay_file_name + '.hkl'
						data = {'home_feature_screen': self.home_feature_screen, 
								 'home_feature_minimap': self.home_feature_minimap, 
								 'home_player': self.home_player,
								 'home_feature_units': self.home_feature_units,
								 'home_game_loop': self.home_game_loop,
								 'home_available_actions': self.home_available_actions,
								 'home_action': self.home_action
								 }

						self.home_feature_screen = []
						self.home_feature_minimap = []
						self.home_player = []
						self.home_feature_units = []
						self.home_game_loop = []
						self.home_available_actions = []

						self.home_action = []

						hkl.dump(data, file_path)
						break

					_state = StepType.MID

				#self.home_BO = build_info
				#self.away_BU = score_cumulative_dict
			except:
				continue


replay = Trajectory(arguments.replay_path, arguments.player_1, arguments.player_2, arguments.mmr)
replay.generate_trajectory()