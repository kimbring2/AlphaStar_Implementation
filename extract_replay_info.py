#!/usr/bin/env python
from pysc2.lib import features, point, actions, units
from pysc2.env import sc2_env, available_actions_printer
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

import importlib
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("replay", None, "Path to a replay file.")
flags.mark_flag_as_required("replay")

function_dict = {}
for _FUNCTION in actions._FUNCTIONS:
    print(_FUNCTION)
    function_dict[_FUNCTION.ability_id] = _FUNCTION.name

class ReplayEnv:
    def __init__(self,
                 replay_file_path,
                 player_id=1,
                 screen_size_px=(64, 64),
                 minimap_size_px=(64, 64),
                 discount=1.,
                 step_mul=1):

        self.discount = discount
        self.step_mul = step_mul

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(replay_file_path)
        self.ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        #print("self.info: " + str(self.info))

        self.base_build = (self.info).base_build
        self.map_name = (self.info).map_name
        self.game_duration_loops = (self.info).game_duration_loops

        #print("(self.info).player_info[0].player_result.result: " + str((self.info).player_info[0].player_result.result))
        self.player0_race = (self.info).player_info[0].player_info.race_actual
        self.player0_mmr = (self.info).player_info[0].player_mmr
        self.player0_apm = (self.info).player_info[0].player_apm
        self.player0_result = (self.info).player_info[0].player_result.result
        print("self.player0_race: " + str(self.player0_race))
        print("self.player0_mmr: " + str(self.player0_mmr))
        print("self.player0_apm: " + str(self.player0_apm))
        print("self.player0_result: " + str(self.player0_result))

        #print("(self.info).player_info[1].player_result.result: " + str((self.info).player_info[1].player_result.result))
        self.player1_race = (self.info).player_info[1].player_info.race_actual
        self.player1_mmr = (self.info).player_info[1].player_mmr
        self.player1_apm = (self.info).player_info[1].player_apm
        self.player1_result = (self.info).player_info[1].player_result.result
        print("self.player1_race: " + str(self.player1_race))
        print("self.player1_mmr: " + str(self.player1_mmr))
        print("self.player1_apm: " + str(self.player1_apm))
        print("self.player1_result: " + str(self.player1_result))
        
        if not self._valid_replay(self.info, self.ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if (self.info).local_map_path:
            map_data = self.run_config.map_data((self.info).local_map_path)

        self._episode_length = (self.info).game_duration_loops
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
        #   for p in info.player_info:
        #       if p.player_apm < 10 or p.player_mmr < 1000:
        #           # Low APM = player just standing around.
        #           # Low MMR = corrupt replay or player who is weak.
        #           return False
        return True

    def start(self):
        feature_screen_size = 64
        feature_minimap_size = 64
        rgb_screen_size = None
        rgb_minimap_size = None
        action_space = None
        use_feature_units = True
        agent_interface_format = sc2_env.parse_agent_interface_format(
          feature_screen=feature_screen_size,
          feature_minimap=feature_minimap_size,
          rgb_screen=rgb_screen_size,
          rgb_minimap=rgb_minimap_size,
          action_space=action_space,
          use_feature_units=use_feature_units)

        _features = features.features_from_game_info(self.controller.game_info(), agent_interface_format=agent_interface_format)

        result_list = ['Win', 'Loss']
        race_list = ['Terran', 'Zerg', 'Protoss']
        act_list = []
        obs_list = []
        while True:
        #for i in range(0, 1000):
            self.controller.step(self.step_mul)
            obs = self.controller.observe()

            #print("type(obs.actions): " + str(type(obs.actions)))
            #print("len(obs.actions): " + str(len(obs.actions)))
            if (len(obs.actions) != 0):
                action = (obs.actions)[0]
                #print("action: " + str(action))

                action_spatial = action.action_feature_layer
                #print("type(action.action_feature_layer): " + str(type(action.action_feature_layer)))
                #print("action: " + str(action))
                #print("type(action_spatial): " + str(type(action_spatial)))
                #print("action_spatial: " + str(action_spatial))
                
                unit_command = action_spatial.unit_command
                ability_id = unit_command.ability_id
                #print("unit_command.ability_id: " + str(unit_command.ability_id))

                function_name = function_dict[ability_id]
                if (function_name != 'build_queue'):
                    function_name_first = str(function_name[0:5])
                    #print("function_name_first: " + str(function_name_first))

                    if (function_name_first == 'Build' or function_name_first == 'Train'):
                        print("function_name: " + str(function_name))

                #print("unit_command.target_screen_coord: " + str(unit_command.target_screen_coord))
                #print("unit_command.target_minimap_coord: " + str(unit_command.target_minimap_coord))
                parsed_unit_command = {"ability_id":str(unit_command.ability_id), "target_screen_coord":str(unit_command.target_screen_coord), 
                                       "target_minimap_coord":str(unit_command.target_minimap_coord)}

                camera_move = action_spatial.camera_move
                #print("camera_move.center_minimap.x: " + str(camera_move.center_minimap.x))
                #print("camera_move.center_minimap.y: " + str(camera_move.center_minimap.y))
                parsed_camera_move = {"center_minimap_x":str(camera_move.center_minimap.x), "center_minimap_y":str(camera_move.center_minimap.y)}

                unit_selection_point = action_spatial.unit_selection_point
                #print("unit_selection_point.selection_screen_coord: " + str(unit_selection_point.selection_screen_coord))
                #print("unit_selection_point.Type: " + str(unit_selection_point.Type))
                parsed_unit_selection_point = {"selection_screen_coord":str(unit_selection_point.selection_screen_coord), 'Type':str(unit_selection_point.Type)}

                unit_selection_rect = action_spatial.unit_selection_rect
                #print("unit_selection_rect.selection_screen_coord: " + str(unit_selection_rect.selection_screen_coord))
                #print("unit_selection_rect.selection_add: " + str(unit_selection_rect.selection_add))
                parsed_unit_selection_rect = {"selection_screen_coord":str(unit_selection_rect.selection_screen_coord), 'selection_add':str(unit_selection_rect.selection_add)}
                
                parsed_action = {"parsed_unit_command":parsed_unit_command, "parsed_camera_move":parsed_camera_move, 
                                 "parsed_unit_selection_point":parsed_unit_selection_point, "parsed_unit_selection_rect":parsed_unit_selection_rect}
                #print("action.action_feature_layer: " + str(action.action_feature_layer))
                #print("")
                #act_list.append(parsed_action)

            agent_obs = _features.transform_obs(obs)
            feature_units = agent_obs['feature_units']

            unit_list = []
            for feature_unit in feature_units:
                unit_type = feature_unit.unit_type
                owner = feature_unit.owner

                if owner == 1:
                    #print("unit_type: " + str(unit_type))
                    unit_list.append(unit_type)

            #print("len(unit_list): " + str(len(unit_list)))
            #print("unit_list: " + str(unit_list))
            #obs_list.append(agent_obs)
            #print("type(obs.actions): " + str(type(obs.actions)))
            #print("type(agent_obs): " + str(type(agent_obs)))
            #print("type(obs.actions): " + str(type(obs.actions)))
            #print("agent_obs.keys(): " + str(agent_obs.keys()))

            # ['unit_type', 'player_relative', 'health', 'shields', 'energy', 'transport_slots_taken', 'build_progress']
            #print("agent_obs.keys(): " + str(agent_obs.keys()))
            #print("agent_obs: " + str(agent_obs))
            build_queue = agent_obs['build_queue']
            #print("build_queue: " + str(build_queue))
            if len(build_queue) != 0:
                #print("build_queue: " + str(build_queue))

                build_progress = build_queue[0][6]

                #if build_progress == 99:
                    #print("build_queue: " + str(build_queue))
                    #print("build_progress: " + str(build_progress))

            # [None, ['ability_id', 'build_progress']]
            production_queue = agent_obs['production_queue']
            #print("production_queue: " + str(production_queue))
            #if len(production_queue) != 0:
                #print("production_queue[0][0]: " + str(production_queue[0][0]))
                #if production_queue[0][0] == 881:
                    #print("production_queue: " + str(production_queue))
            #print("")

            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += self.step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)
            #print("obs.actions: " + str(obs.actions))
            #self.agent.step(step, obs.actions)

            if obs.player_result:
                break

            self._state = StepType.MID

        data_dict = {'observation' : obs_list, 'action' : act_list}
        #data_dict = {'observation' : obs_list}

        player0_txt = race_list[self.player0_race - 1] + '(mmr:' + str(self.player0_mmr) + ',' + result_list[self.player0_result - 1] + ')'
        player1_txt = race_list[self.player1_race - 1] + '(mmr:' + str(self.player1_mmr) + ',' + result_list[self.player1_result - 1] + ')'
        file_index = 0
        file_name = str(file_index) + '_' + player0_txt + 'vs' + player1_txt
        #print("save file")
        #np.save(file_name + ".npy", data_dict)

        '''
        data = np.load("0_Protoss(mmr:2508,Loss)vsZerg(mmr2516,Win).npy")
        data.item().get('observation')
        '''

def main(unused):
    Replay_Environment = ReplayEnv(FLAGS.replay)
    Replay_Environment.start()

if __name__ == "__main__":
    app.run(main)
