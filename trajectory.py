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

#from absl import app, flags
#FLAGS = flags.FLAGS
#FLAGS(sys.argv)


class Trajectory(object):
	def __init__(self, source, home_race_name, away_race_name, replay_filter, filter_repeated_camera_moves=True):
	    self.source = source
	    self.home_race_name = home_race_name
	    self.away_race_name = away_race_name
	    self.replay_filter = replay_filter
	    self.filter_repeated_camera_moves = filter_repeated_camera_moves

	    self.home_BO = None
	    self.home_trajectory = []

	    self.away_BU = None
	    self.away_trajectory = []

	    self.home_score_cumulative = None
	    self.away_score_cumulative = None

	def get_BO(self, player):
		if player == 0:
			return self.home_BO
		else:
			return self.away_BU

	def get_random_trajectory(self):
		function_dict = {}
		for _FUNCTION in actions._FUNCTIONS:
		    #print(_FUNCTION)
		    function_dict[_FUNCTION.ability_id] = _FUNCTION.name

		race_list = ['Terran', 'Zerg', 'Protoss']

		"""How many agent steps the agent has been trained for."""
		run_config = run_configs.get()
		sc2_proc = run_config.start()
		controller = sc2_proc.controller

		#print ("source: {}".format(source))
		#root_path = '/media/kimbring2/Steam/StarCraftII/Replays/4.8.2.71663-20190123_035823-1'
		root_path =  self.source
		file_list = glob.glob(root_path + '*.*')
		#print ("file_list: {}".format(file_list))

		for i in range(0, 500):
			#print("i: " + str(i))

			replay_file_path = random.choice(file_list)
			#print ("replay_file_path: {}".format(replay_file_path))
			#replay_file_path = root_path + '/Simple64_2021-06-28-21-03-54.SC2Replay'
			#replay_file_path = root_path + '/0a1b09abc9e98f4e0c3921ae0a427c27e97c2bbdcf34f50df18dc41cea3f3249.SC2Replay'
			#replay_file_path_2 = root_path + '/0a01d32e9a98e1596b88bc2cdec7752249b22aca774e3305dae2e93efef34be3.SC2Replay'
			#replay_file_path_0 = human_data
			#print ("replay_file_path: {}".format(replay_file_path))
			try: 
				replay_data = run_config.replay_data(replay_file_path)
				ping = controller.ping()
				info = controller.replay_info(replay_data)
				print("ping: " + str(ping))
				print("replay_info: " + str(info))

				player0_race = info.player_info[0].player_info.race_actual
				player0_mmr = info.player_info[0].player_mmr
				player0_apm = info.player_info[0].player_apm
				player0_result = info.player_info[0].player_result.result
				print("player0_race: " + str(player0_race))
				print("player0_mmr: " + str(player0_mmr))
				print("player0_apm: " + str(player0_apm))
				print("player0_result: " + str(player0_result))
				'''
				home_race = race_list.index(self.home_race_name) + 1
				if (home_race == player0_race):
					print("player0_race pass")
				else:
					print("player0_race fail")
					continue

				
				if (player0_mmr >= self.replay_filter):
					print("player0_mmr pass ")
				else:
					print("player0_mmr fail")
					continue
				
				player1_race = info.player_info[0].player_info.race_actual
				player1_mmr = info.player_info[0].player_mmr
				player1_apm = info.player_info[0].player_apm
				player1_result = info.player_info[0].player_result.result
				print("player1_race: " + str(player1_race))
				print("player1_mmr: " + str(player1_mmr))
				print("player1_apm: " + str(player1_apm))
				print("player1_result: " + str(player1_result))

				
				away_race = race_list.index(self.away_race_name) + 1
				if (away_race == player1_race):
					print("player1_race pass ")
				else:
					print("player1_race fail ")
					continue

				if (player1_mmr >= self.replay_filter):
					print("player1_mmr pass ")
				else:
					print("player1_mmr fail")
					continue
				'''
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

				#print("True loop")
				while True:
					replay_step += 1
					#print("#####################################################")
					print("replay_step: " + str(replay_step))

					#if replay_step == 800:
					#	break

					controller.step(step_mul)
					obs = controller.observe()
					#print("obs.actions: " + str(obs.actions))
					#print("len(obs.actions): " + str(len(obs.actions)))
					#print("")
					#self.home_trajectory.append(obs)
					'''
					agent_act = None
					if len(obs.actions) != 0:
						action = (obs.actions)[0]
						action_feature_layer = action.action_feature_layer
						unit_command = action_feature_layer.unit_command
						ability_id = unit_command.ability_id
						function_name = function_dict[ability_id]
						#print("action: " + str(action))
						#print("action_feature_layer: " + str(action_feature_layer))
						#print("unit_command: " + str(unit_command))
						#print("ability_id: " + str(ability_id))
						#print("function_name: " + str(function_name))
						#print("#####################################################")
						#print("")
						if (function_name != 'build_queue'):
							function_name_parse = function_name.split('_')
							function_name_first = function_name_parse[0]
							#print("function_name_first: " + str(function_name_first))
							if (function_name_first == 'Build' or function_name_first == 'Train'):
								unit_name = function_name_parse[1]
								unit_info = int(units_new.get_unit_type(self.home_race_name, unit_name)[0])
								#print("unit_name: " + str(unit_name))
								#print("unit_info: " + str(unit_info))

								#print("function_name_parse[1]: " + str(function_name_parse[1]))
								build_name.append(unit_name)
								build_info.append(unit_info)
					'''

					if obs.player_result: # Episide over.
						_state = StepType.LAST
						discount = 0
					else:
						discount = discount
						_episode_steps += step_mul

					agent_obs = _features.transform_obs(obs)

					exec_actions = []
					if len(obs.actions) != 0:
						for ac in obs.actions:
							#print("ac: " + str(ac))
							exec_act = _features.reverse_action(ac)
							#print("exec_act: " + str(exec_act))
							exec_actions.append(exec_act)
					else:
						exec_act = actions.FUNCTIONS.no_op()
						exec_actions.append(exec_act)
					
					#print("exec_actions: " + str(exec_actions))
					#print("")
					
					self.home_trajectory.append([agent_obs, exec_actions, _state])
					step = TimeStep(step_type=_state, reward=0,
				                       discount=discount, observation=agent_obs)

					#print("obs.player_result: " + str(obs.player_result))
					if obs.player_result:
						break

					_state = StepType.MID

				#self.home_BO = build_info
				#self.away_BU = score_cumulative_dict
				break
			except:
				continue

