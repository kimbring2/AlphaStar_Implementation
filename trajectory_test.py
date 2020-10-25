#!/usr/bin/env python
from pysc2.lib import features, point, actions, units
from pysc2.env.environment import TimeStep, StepType
from pysc2.env import sc2_env, available_actions_printer
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import units_new

import importlib
import random
import sys
import glob

from absl import app, flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

function_dict = {}
for _FUNCTION in actions._FUNCTIONS:
    #print(_FUNCTION)
    function_dict[_FUNCTION.ability_id] = _FUNCTION.name

race_list = ['Terran', 'Zerg', 'Protoss']

# from trajectory import get_random_trajectory
# get_random_trajectory(source='/media/kimbring2/Steam1/StarCraftII/Replays/4.8.2.71663-20190123_035823-1/', home_race=1, away_race=1, replay_filter=3500)
# import sys, importlib
# importlib.reload(sys.modules['trajectory'])

home_race = 1
away_race = 1

run_config = run_configs.get()
sc2_proc = run_config.start()
controller = sc2_proc.controller

#print ("source: {}".format(source))
root_path = '/media/kimbring2/Steam/StarCraftII/Replays/'
#root_path = '/media/kimbring2/Steam'
#root_path = source
file_list = glob.glob(root_path + '*.*')
#print ("file_list: {}".format(file_list))

for i in range(500):
	#print("i: " + str(i))

	replay_file_path = random.choice(file_list)
	#print ("replay_file_path: {}".format(replay_file_path))
	#replay_file_path = '/media/kimbring2/Steam/ffff1bf42548346e8b40de976324bb221351ee8b4712a119400dd84235a92d24.SC2Replay'
	print ("replay_file_path: {}".format(replay_file_path))

	try: 
		replay_data = run_config.replay_data(replay_file_path)
		ping = controller.ping()
		info = controller.replay_info(replay_data)
		print("ping: " + str(ping))
		print("info: " + str(info))

		player0_race = info.player_info[0].player_info.race_actual
		player0_mmr = info.player_info[0].player_mmr
		player0_apm = info.player_info[0].player_apm
		player0_result = info.player_info[0].player_result.result
		print("player0_race: " + str(player0_race))
		print("player0_mmr: " + str(player0_mmr))
		print("player0_apm: " + str(player0_apm))
		print("player0_result: " + str(player0_result))

		home_race_name = 'Terran'
		home_race = race_list.index(home_race_name) + 1
		if (home_race == player0_race):
			print("player0_race pass ")
		else:
			continue

		player1_race = info.player_info[0].player_info.race_actual
		player1_mmr = info.player_info[0].player_mmr
		player1_apm = info.player_info[0].player_apm
		player1_result = info.player_info[0].player_result.result
		print("player1_race: " + str(player1_race))
		print("player1_mmr: " + str(player1_mmr))
		print("player1_apm: " + str(player1_apm))
		print("player1_result: " + str(player1_result))

		print("away_race: " + str(away_race))
		print("player1_race: " + str(player1_race))
		if (away_race == player1_race):
			print("player1_race pass ")
		else:
			continue

		replay_filter = 2500
		if (player0_mmr >= replay_filter):
			print("player0_mmr pass ")
		else:
			continue
		screen_size_px = (128, 128)
		minimap_size_px = (64, 64)
		player_id = 1
		discount = 1.
		step_mul = 8

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
			print("")
			continue

		feature_screen_size = 128
		feature_minimap_size = 64
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

		print("controller.game_info(): " + str(controller.game_info()))
		_features = features.features_from_game_info(controller.game_info(), agent_interface_format=aif)

		build_name = []
		build_info = []
		replay_step = 0
		while True:
			replay_step += 1

			controller.step(step_mul)
			obs = controller.observe()
			agent_obs = _features.transform_obs(obs)
			print("replay_step: " + str(replay_step))
			#print("agent_obs['feature_units']: " + str(agent_obs['feature_units']))
			#print("len(obs.actions): " + str(len(obs.actions)))
			
			if (len(obs.actions) != 0):
				action = (obs.actions)[0]
				#print("action: " + str(action))

				action_spatial = action.action_feature_layer
				unit_command = action_spatial.unit_command
				ability_id = unit_command.ability_id
				function_name = function_dict[ability_id]

				#print("function_name: " + str(function_name))
				
				if (function_name != 'build_queue'):
					function_name_parse = function_name.split('_')
					#print("function_name_parse: " + str(function_name_parse))

					function_name_first = function_name_parse[0]
					#print("function_name_first: " + str(function_name_first))
					
					if (function_name_first == 'Build' or function_name_first == 'Train'):
						unit_name = function_name_parse[1]
						print("unit_name: " + str(unit_name))

						unit_info = int(units_new.get_unit_type(home_race_name, unit_name)[0])
						print("unit_info: " + str(unit_info))

						build_name.append(unit_name)
						build_info.append(unit_info)
					
			if obs.player_result: # Episide over.
				_state = StepType.LAST
				discount = 0
			else:
				discount = discount
				_episode_steps += step_mul

			agent_obs = _features.transform_obs(obs)
			#print("agent_obs['feature_units']: " + str(agent_obs['feature_units']))

			step = TimeStep(step_type=_state, reward=0,
			                   discount=discount, observation=agent_obs)

			score_cumulative = agent_obs['score_cumulative']
			score_cumulative_dict = {}
			score_cumulative_dict['score'] = score_cumulative.score
			score_cumulative_dict['idle_production_time'] = score_cumulative.idle_production_time
			score_cumulative_dict['idle_worker_time'] = score_cumulative.idle_worker_time
			score_cumulative_dict['total_value_units'] = score_cumulative.total_value_units
			score_cumulative_dict['total_value_structures'] = score_cumulative.total_value_structures
			score_cumulative_dict['killed_value_units'] = score_cumulative.killed_value_units
			score_cumulative_dict['killed_value_structures'] = score_cumulative.killed_value_structures
			score_cumulative_dict['collected_minerals'] = score_cumulative.collected_minerals
			score_cumulative_dict['collected_vespene'] = score_cumulative.collected_vespene
			score_cumulative_dict['collection_rate_minerals'] = score_cumulative.collection_rate_minerals
			score_cumulative_dict['collection_rate_vespene'] = score_cumulative.collection_rate_vespene
			score_cumulative_dict['spent_minerals'] = score_cumulative.spent_minerals
			score_cumulative_dict['spent_vespene'] = score_cumulative.spent_vespene

			if obs.player_result:
				break

			_state = StepType.MID
			
		#print("build_name: " + str(build_name))
		#print("build_info: " + str(build_info))
		#print("score_cumulative_dict: " + str(score_cumulative_dict))
		#print("")
		#return build_info, score_cumulative_dict
	except:
		print("except")
		continue

