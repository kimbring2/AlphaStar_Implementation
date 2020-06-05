#!/usr/bin/env python

from pysc2.lib import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib

import sys

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

run_config = run_configs.get()
sc2_proc = run_config.start()
controller = sc2_proc.controller

root_path = '/media/kimbring2/Steam1/StarCraftII/Replays/4.8.2.71663-20190123_035823-1/'
replay_file_path_0 = root_path + '/0a0f62052fe4311368910ad38c662bf979e292b86ad02b49b41a87013e58c432.SC2Replay'
replay_file_path_1 = root_path + '/0a1b09abc9e98f4e0c3921ae0a427c27e97c2bbdcf34f50df18dc41cea3f3249.SC2Replay'
replay_file_path_2 = root_path + '/0a01d32e9a98e1596b88bc2cdec7752249b22aca774e3305dae2e93efef34be3.SC2Replay'

replay_data = run_config.replay_data(replay_file_path_0)
ping = controller.ping()
info = controller.replay_info(replay_data)

print("ping: " + str(ping))
print("info: " + str(info))

screen_size_px = (128, 128)
minimap_size_px = (64, 64)
player_id = 1
discount = 1.
step_mul = 1

screen_size_px = point.Point(*screen_size_px)
minimap_size_px = point.Point(*minimap_size_px)
interface = sc_pb.InterfaceOptions(raw=False, score=True,
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

_features = features.features_from_game_info(controller.game_info())

while True:
	controller.step(step_mul)
	obs = controller.observe()
	try:
		agent_obs = _features.transform_obs(obs)
		#print("agent_obs.keys(): " + str(agent_obs.keys()))

		# ['unit_type', 'player_relative', 'health', 'shields', 'energy', 'transport_slots_taken', 'build_progress']
		#print("agent_obs.keys(): " + str(agent_obs.keys()))
		#print("agent_obs: " + str(agent_obs))
		#print("agent_obs['build_queue']: " + str(agent_obs['build_queue']))

		# [None, ['ability_id', 'build_progress']]
		#print("agent_obs['production_queue']: " + str(agent_obs['production_queue']))
		#print("")
	except:
		pass

	if obs.player_result: # Episide over.
		_state = StepType.LAST
		discount = 0

	else:
		discount = discount

		_episode_steps += step_mul

	step = TimeStep(step_type=_state, reward=0,
                    discount=discount, observation=agent_obs)

	print("step: " + str(step))
	#agent.step(step, obs.actions)

	if obs.player_result:
		break

	_state = StepType.MID