from pysc2.lib import actions, features, units
import numpy as np
import units_new
import upgrades_new
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf


terran_building_list = ['Armory', 'AutoTurret', 'Barracks', 'BarracksFlying', 'BarracksReactor', 'BarracksTechLab', 
                             'Bunker', 'CommandCenter', 'CommandCenterFlying', 'EngineeringBay', 'Factory', 'FactoryFlying',
                             'FactoryReactor', 'FactoryTechLab', 'FusionCore', 'GhostAcademy', 'MissileTurret', 'OrbitalCommand',
                             'OrbitalCommandFlying', 'PlanetaryFortress', 'Reactor', 'Refinery', 'RefineryRich', 'SensorTower', 
                             'Starport', 'StarportFlying', 'StarportReactor', 'StarportTechLab', 'SupplyDepot', 'SupplyDepotLowered',
                             'TechLab'] 
terran_air_unit_list = ['Banshee', 'Battlecruiser', 'Cyclone', 'Liberator', 'LiberatorAG', 'Medivac', 'PointDefenseDrone', 'Raven',
                             'VikingAssault']
terran_ground_unit_list = ['Ghost', 'GhostAlternate', 'GhostNova', 'Hellion', 'Hellbat', 'KD8Charge', 'MULE', 'Marauder', 'Marine',
                                 'Nuke', 'PointDefenseDrone']


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

# Action part
_NO_OP = [actions.FUNCTIONS.no_op]

_MOVE_SCREEN = [actions.FUNCTIONS.Move_screen]
_MOVE_CAMERA = [actions.FUNCTIONS.move_camera]
_HOLDPOSITION_QUICK = [actions.FUNCTIONS.HoldPosition_quick]
_SELECT_ARMY = [actions.FUNCTIONS.select_army]

_SELECT_POINT_SELECT = [actions.FUNCTIONS.select_point, actions.SelectPointAct.select]
_SELECT_POINT_TOGGLE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.toggle]
_SELECT_POINT_SELECT_ALL_TYPE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.select_all_type]
_SELECT_POINT_ADD_ALL_TYPE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.add_all_type]

_SELECT_RECT_SELECT = [actions.FUNCTIONS.select_rect, actions.SelectAdd.select]
_SELECT_RECT_ADD = [actions.FUNCTIONS.select_rect, actions.SelectAdd.add]

_SELECT_IDLE_WORKER_SELECT = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.select]
_SELECT_IDLE_WORKER_ADD = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.add]
_SELECT_IDLE_WORKER_SELECT_ALL = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.select_all]
_SELECT_IDLE_WORKER_ADD_ALL = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.add_all]

_SELECT_CONTROL_GROUP_RECALL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.recall]
_SELECT_CONTROL_GROUP_SET = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.set]
_SELECT_CONTROL_GROUP_APPEND = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.append]
_SELECT_CONTROL_GROUP_SET_AND_STEAL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.set_and_steal]
_SELECT_CONTROL_GROUP_APPEND_AND_STEAL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.append_and_steal]

_SELECT_UNIT_SELECT = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.select]
_SELECT_UNIT_DESELECT = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.deselect]
_SELECT_UNIT_SELECT_ALL_TYPE = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.select_all_type]
_SELECT_UNIT_DESELECT_ALL_TYPE = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.deselect_all_type]

_SMART_SCREEN = [actions.FUNCTIONS.Smart_screen]
_SMART_MINIMAP = [actions.FUNCTIONS.Smart_minimap]

_ATTACK_SCREEN = [actions.FUNCTIONS.Attack_screen]
_ATTACK_MINIMAP = [actions.FUNCTIONS.Attack_minimap]

_BUILD_COMMANDCENTER_SCREEN = [actions.FUNCTIONS.Build_CommandCenter_screen]
_BUILD_SUPPLYDEPOT_SCREEN = [actions.FUNCTIONS.Build_SupplyDepot_screen]
_BUILD_BARRACKS_SCREEN = [actions.FUNCTIONS.Build_Barracks_screen]
_BUILD_REFINERY_SCREEN = [actions.FUNCTIONS.Build_Refinery_screen]
_BUILD_TECHLAB_SCREEN = [actions.FUNCTIONS.Build_TechLab_screen]
_BUILD_TECHLAB_QUICK = [actions.FUNCTIONS.Build_TechLab_quick]
_BUILD_REACTOR_QUICK = [actions.FUNCTIONS.Build_Reactor_quick]
_BUILD_REACTOR_SCREEN = [actions.FUNCTIONS.Build_Reactor_screen]
_BUILD_BUNKER_SCREEN = [actions.FUNCTIONS.Build_Bunker_screen]
_BUILD_STARPORT_SCREEN = [actions.FUNCTIONS.Build_Starport_screen]
_BUILD_FACTORY_SCREEN = [actions.FUNCTIONS.Build_Factory_screen]
_BUILD_ARMORY_SCREEN = [actions.FUNCTIONS.Build_Armory_screen]
_BUILD_ENGINNERINGBAY_SCREEN = [actions.FUNCTIONS.Build_EngineeringBay_screen]

_TRAIN_MARINE_QUICK = [actions.FUNCTIONS.Train_Marine_quick]
_TRAIN_MARAUDER_QUICK = [actions.FUNCTIONS.Train_Marauder_quick]
_TRAIN_SCV_QUICK = [actions.FUNCTIONS.Train_SCV_quick]
_TRAIN_SIEGETANK_QUICK = [actions.FUNCTIONS.Train_SiegeTank_quick]
_TRAIN_MEDIVAC_QUICK = [actions.FUNCTIONS.Train_Medivac_quick]
_TRAIN_REAPER_QUICK = [actions.FUNCTIONS.Train_Reaper_quick]
_TRAIN_HELLION_QUICK = [actions.FUNCTIONS.Train_Hellion_quick]
_TRAIN_VIKINGFIGHTER_QUICK = [actions.FUNCTIONS.Train_VikingFighter_quick]

_RETURN_SCV_QUICK = [actions.FUNCTIONS.Harvest_Return_SCV_quick]
_HARVEST_GATHER_SCREEN = [actions.FUNCTIONS.Harvest_Gather_screen]
_HARVEST_GATHER_SCV_SCREEN = [actions.FUNCTIONS.Harvest_Gather_SCV_screen]

_LIFT_QUICK = [actions.FUNCTIONS.Lift_quick]
_MORPH_SUPPLYDEPOT_LOWER_QUICK = [actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick]
_MORPH_SUPPLYDEPOT_RAISE_QUICK = [actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick]
_MORPH_ORBITALCOMMAND_QUICK = [actions.FUNCTIONS.Morph_OrbitalCommand_quick]
_LAND_SCREEN = [actions.FUNCTIONS.Land_screen]
_CANCEL_LAST_QUICK = [actions.FUNCTIONS.Cancel_Last_quick]
_RALLY_WORKERS_SCREEN = [actions.FUNCTIONS.Rally_Workers_screen]
_HARVEST_RETURN_QUICK = [actions.FUNCTIONS.Harvest_Return_quick]
_PATROL_SCREEN = [actions.FUNCTIONS.Patrol_screen]
_EFFECT_COOLDOWNMULE_SCREEN = [actions.FUNCTIONS.Effect_CalldownMULE_screen]
_BUILD_QUEUE = [actions.FUNCTIONS.build_queue]
_EFFECT_KD8CHARGE_SCREEN = [actions.FUNCTIONS.Effect_KD8Charge_screen]
_EFFECT_SPRAY_SCREEN = [actions.FUNCTIONS.Effect_Spray_screen]
_HALT_QUICK = [actions.FUNCTIONS.Halt_quick]

_RESEARCH_STIMPACK_QUICK = [actions.FUNCTIONS.Research_Stimpack_quick]
_RESEARCH_COMBATSHIELD_QUICK = [actions.FUNCTIONS.Research_CombatShield_quick]
_UNLOAD = [actions.FUNCTIONS.unload]

action_type_list = [_NO_OP, _BUILD_SUPPLYDEPOT_SCREEN, _BUILD_BARRACKS_SCREEN, _BUILD_REFINERY_SCREEN, _BUILD_TECHLAB_SCREEN, _BUILD_COMMANDCENTER_SCREEN, 
                        _BUILD_REACTOR_QUICK, _BUILD_BUNKER_SCREEN, _BUILD_STARPORT_SCREEN, _BUILD_FACTORY_SCREEN, _HALT_QUICK, _RESEARCH_COMBATSHIELD_QUICK,
                        _TRAIN_MARINE_QUICK, _TRAIN_MARAUDER_QUICK, _TRAIN_SCV_QUICK, _TRAIN_SIEGETANK_QUICK, _TRAIN_MEDIVAC_QUICK, _SELECT_UNIT_DESELECT, 
                        _SELECT_UNIT_SELECT_ALL_TYPE, _SELECT_UNIT_DESELECT_ALL_TYPE, _TRAIN_REAPER_QUICK,
                        _RETURN_SCV_QUICK, _HARVEST_GATHER_SCREEN, _HARVEST_GATHER_SCV_SCREEN, _PATROL_SCREEN, _SELECT_UNIT_SELECT, _HOLDPOSITION_QUICK,
                        _LIFT_QUICK, _MORPH_SUPPLYDEPOT_LOWER_QUICK, _LAND_SCREEN, _BUILD_TECHLAB_QUICK, 
                        _RESEARCH_STIMPACK_QUICK, _SELECT_POINT_SELECT, _SELECT_POINT_TOGGLE, _SELECT_POINT_SELECT_ALL_TYPE, _SELECT_POINT_ADD_ALL_TYPE,
                        _ATTACK_SCREEN, _ATTACK_MINIMAP, _SMART_SCREEN, _SMART_MINIMAP, _MORPH_ORBITALCOMMAND_QUICK, _BUILD_ENGINNERINGBAY_SCREEN,
                        _SELECT_RECT_SELECT, _SELECT_RECT_ADD, _SELECT_IDLE_WORKER_SELECT, _SELECT_IDLE_WORKER_SELECT_ALL, _SELECT_IDLE_WORKER_ADD_ALL, 
                        _SELECT_IDLE_WORKER_ADD, _SELECT_CONTROL_GROUP_RECALL, _SELECT_CONTROL_GROUP_SET, _SELECT_CONTROL_GROUP_APPEND, _SELECT_CONTROL_GROUP_SET_AND_STEAL,
                        _SELECT_CONTROL_GROUP_APPEND_AND_STEAL, _SELECT_ARMY, _BUILD_ARMORY_SCREEN, _BUILD_REACTOR_SCREEN,
                        _MOVE_SCREEN, _MOVE_CAMERA, _CANCEL_LAST_QUICK, _RALLY_WORKERS_SCREEN, _HARVEST_RETURN_QUICK, _TRAIN_HELLION_QUICK, 
                        _EFFECT_COOLDOWNMULE_SCREEN, _MORPH_SUPPLYDEPOT_RAISE_QUICK, _BUILD_QUEUE, _EFFECT_KD8CHARGE_SCREEN, _UNLOAD, _EFFECT_SPRAY_SCREEN,
                        _TRAIN_VIKINGFIGHTER_QUICK, _SELECT_POINT_SELECT, _SELECT_POINT_TOGGLE, _SELECT_POINT_SELECT_ALL_TYPE, _SELECT_POINT_ADD_ALL_TYPE]

action_id_list = []
for action_type in action_type_list:

  if len(action_type) != 1:
    action_id_list.append([int(action_type[0].id), int(action_type[1])])
  else:
    action_id_list.append([int(action_type[0].id)])

action_len = len(action_type_list)


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


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
  time = get_gameloop_obs(game_loop)

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

'''
FunctionCall(function=<_Functions.no_op: 0>, arguments=[])

FunctionCall(function=<_Functions.move_camera: 1>, arguments=[[49, 18]])

[FunctionCall(function=<_Functions.build_queue: 11>, arguments=[[0]])

FunctionCall(function=<_Functions.Smart_screen: 451>, arguments=[[<Queued.now: 0>], [94, 56]])
FunctionCall(function=<_Functions.Smart_minimap: 452>, arguments=[[<Queued.now: 0>], [51, 19]])
FunctionCall(function=<_Functions.Attack_screen: 12>, arguments=[[<Queued.now: 0>], [85, 54]])
FunctionCall(function=<_Functions.Attack_minimap: 13>, arguments=[[<Queued.now: 0>], [51, 20]])
FunctionCall(function=<_Functions.Effect_CalldownMULE_screen: 183>, arguments=[[<Queued.now: 0>], [24, 56]])

FunctionCall(function=<_Functions.select_control_group: 4>, arguments=[[<ControlGroupAct.recall: 0>], [4]])
FunctionCall(function=<_Functions.select_control_group: 4>, arguments=[[<ControlGroupAct.set: 1>], [1]]

FunctionCall(function=<_Functions.select_point: 2>, arguments=[[<SelectPointAct.select: 0>], [69, 64]])

FunctionCall(function=<_Functions.select_unit: 5>, arguments=[[<SelectUnitAct.select: 0>], [1]]

FunctionCall(function=<_Functions.select_rect: 3>, arguments=[[<SelectAdd.add: 1>], [79, 11], [124, 49]]

FunctionCall(function=<_Functions.Build_SupplyDepot_screen: 91>, arguments=[[<Queued.now: 0>], [79, 28]])
FunctionCall(function=<_Functions.Build_CommandCenter_screen: 44>, arguments=[[<Queued.now: 0>], [66, 28]])
FunctionCall(function=<_Functions.Build_Refinery_screen: 79>, arguments=[[<Queued.now: 0>], [116, 73]])
FunctionCall(function=<_Functions.Build_Barracks_screen: 42>, arguments=[[<Queued.now: 0>], [66, 41]])

FunctionCall(function=<_Functions.select_army: 7>, arguments=[[<SelectAdd.select: 0>]])

FunctionCall(function=<_Functions.Train_SCV_quick: 490>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Train_Marine_quick: 477>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Train_SiegeTank_quick: 492>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Morph_SupplyDepot_Lower_quick: 318>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Morph_OrbitalCommand_quick: 309>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Morph_SiegeMode_quick: 317>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Morph_Unsiege_quick: 322>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Cancel_Last_quick: 168>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Research_CombatShield_quick: 361>, arguments=[[<Queued.now: 0>]])
FunctionCall(function=<_Functions.Stop_quick: 453>, arguments=[[<Queued.now: 0>]])
'''
def get_action_from_prediction(agent, observation, action_type_index, selected_units, target_unit, screen_target_location_x, screen_target_location_y, 
                                      minimap_target_location_x, minimap_target_location_y):
  #print("action_type: " + str(action_type))
  #print("selected_units: " + str(selected_units))
  #print("target_unit: " + str(target_unit))
  #print("target_location_x: " + str(target_location_x))
  #print("target_location_y: " + str(target_location_y))

  feature_units = observation['feature_units']
  available_actions = observation['available_actions']
  
  action_types = action_type_list[action_type_index[0]]

  action_type_name = None
  action_type_argu = None
  if len(action_types) != 1:
    action_type_name = action_types[0]
    action_type_argu = action_types[1]
  else:
    action_type_name = action_types[0]

  #print("action_type: " + str(action_type))
  #print("action_type.id: " + str(action_type.id))
  #print("action_type.name: " + str(action_type.name))
  #print("action_type.ability_id: " + str(action_type.ability_id))
  #print("action_type.general_id: " + str(action_type.general_id))
  #print("action_type.function_type: " + str(action_type.function_type))
  # action_type.function_type: <function cmd_screen at 0x7fee91fa8ae8>
  # action_type.function_type: <function cmd_quick at 0x7fee91fa8a60>

  argument = []
  #print("action_type.args: " + str(action_type.args))
  for action_type_arg in action_type_name.args:
    #print("action_type_arg: " + str(action_type_arg))
    #print("action_type_arg.id: " + str(action_type_arg.id))
    #print("action_type_arg.name: " + str(action_type_arg.name))
    #print("action_type_arg.sizes: " + str(action_type_arg.sizes))
    #print("action_type_arg.fn: " + str(action_type_arg.fn))
    #print("action_type_arg.count: " + str(action_type_arg.count))
    
    if action_type_arg.id == 0:
      # action_type_arg.name: screen
      # action_type_arg.sizes: (0, 0)
      argument.append([screen_target_location_x[0], screen_target_location_y[0]])
    elif action_type_arg.id == 1:
      # action_type_arg.name: minimap
      # action_type_arg.sizes: (0, 0)
      argument.append([minimap_target_location_x[0], minimap_target_location_y[0]])
    elif action_type_arg.id == 2:
      # action_type_arg.name: screen2
      # action_type_arg.sizes: (0, 0)
      argument.append([screen_target_location_x[0], screen_target_location_y[0]])
    elif action_type_arg.id == 3:
      # action_type_arg.name: queued
      # action_type_arg.sizes: (2,)
      act_name = 'now'
      if act_name == 'now':
        argument.append([0])
      elif act_name == 'queued':
        argument.append([1])
    elif action_type_arg.id == 4:
      # action_type_arg.name: control_group_act
      # action_type_arg.sizes: (5,)
      if action_type_argu is not None:
        argument.append([action_type_argu])
    elif action_type_arg.id == 5:
      # action_type_arg.name: control_group_id
      # action_type_arg.sizes: (10,)
      argument.append(selected_units)
    elif action_type_arg.id == 6:
      # action_type_arg.name: select_point_act
      # action_type_arg.sizes: (4,)
      if action_type_argu is not None:
        argument.append([action_type_argu])
    elif action_type_arg.id == 7:
      # action_type_arg.name: select_add
      # action_type_arg.sizes: (2,)
      if action_type_argu is not None:
        argument.append([action_type_argu])
    elif action_type_arg.id == 8:
      # action_type_arg.name: select_unit_act
      # action_type_arg.sizes: (4,)
      if action_type_argu is not None:
        argument.append([action_type_argu])
    elif action_type_arg.id == 9:
      # action_type_arg.name: select_unit_id
      # action_type_arg.sizes: (500,)
      argument.append(selected_units)
    elif action_type_arg.id == 10:
      # action_type_arg.name: select_worker
      # action_type_arg.sizes: (4,)
      if action_type_argu is not None:
        argument.append([action_type_argu])
    elif action_type_arg.id == 11:
      # action_type_arg.name: build_queue_id
      # action_type_arg.sizes: (10,)
      argument.append(selected_units)
    elif action_type_arg.id == 12:
      # action_type_arg.name: unload_id
      # action_type_arg.sizes: (500,)
      argument.append(selected_units)
  
    #print("")
    #print("")
  
  #action = [actions.FUNCTIONS.no_op()]
  #print("action_type[0]: " + str(action_type[0]))
  #print("len(action_type_list): " + str(len(action_type_list)))
  #print("action_type_list[action_type[0]]: " + str(action_type_list[action_type[0]]))
  if action_type_name.id in available_actions:
    action = [actions.FunctionCall(action_type_name.id, argument)]
    agent.previous_action = action 
  else:
    action = [actions.FUNCTIONS.no_op()]

  #print("action: " + str(action))
  #print("")
  '''
  if action_type[0] == 0:
      action = action
  else:
    selectable_entity_mask = np.zeros(512)
    for idx, feature_unit in enumerate(feature_units):
        selectable_entity_mask[idx] = 1

    if (selected_units < len(feature_units)):
      agent.selected_unit.append(feature_units[selected_units[0]])
    else:
      selected_units = None

    if (target_unit < len(feature_units)):
      target_unit = target_unit[0]
    else:
      target_unit = None

    if agent.action_phase == 0 and selected_units is not None and (_SELECT_POINT in available_actions):
      selected_units_info = feature_units[selected_units[0]]
      select_point = [selected_units_info.x, selected_units_info.y]
      action = [actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, select_point])]
      agent.action_phase = 1
    elif agent.action_phase == 1 and action_type_list[action_type[0]].id in available_actions:
      position = (target_location_x, target_location_y)
      action = [actions.FunctionCall(action_type_list[action_type[0]].id, [_NOT_QUEUED, position])]

    agent.previous_action = action 
    '''
  return action


def get_entity_obs(feature_units):
    unit_type = []
    #unit_attributes = []
    alliance = []
    current_health = []
    current_shields = []
    current_energy = []
    cargo_space_used = []
    cargo_space_maximum = []
    build_progress = []
    current_health_ratio = []
    current_shield_ratio = []
    current_energy_ratio = []
    display_type = []
    x_position = []
    y_position = []
    is_cloaked = []
    is_powered = []
    is_hallucination = []
    is_active = []
    is_on_screen = []
    is_in_cargo = []
    current_minerals = []
    current_vespene = []
    #mined_minerals = []
    #mined_vespene = []
    assigned_harvesters = []
    ideal_harvesters = []
    weapon_cooldown = []
    order_queue_length = []
    order_1 = []
    order_2 = []
    order_3 = []
    order_4 = []
    buffs = []
    addon_type = []
    order_progress_1 = []
    order_progress_2 = []
    weapon_upgrades = []
    armor_upgrades = []
    shield_upgrades = []
    is_selected = []
    #was_targeted = []
    #print("len(feature_units): " + str(len(feature_units)))
    skip_num = 0
    for unit in feature_units:
      unit_info = str(units.get_unit_type(unit.unit_type))
      unit_info = unit_info.split(".")
      unit_race = unit_info[0]
      unit_name = unit_info[1]

      #print("unit_race: " + str(unit_race))
      #print("unit_name: " + str(unit_name))
      #print("units_new.get_unit_type(unit_race, unit_name): " + str(units_new.get_unit_type(unit_race, unit_name)))
      if unit_race != "Terran" and unit_race != "Neutral":
        skip_num += 1
        continue

      #print("units_new.get_unit_type(unit_race, unit_name): " + str(units_new.get_unit_type(unit_race, unit_name)))
      unit_info = int(units_new.get_unit_type(unit_race, unit_name)[0])
      unit_info_onehot = np.identity(256)[unit_info:unit_info+1]
      unit_type.append(unit_info_onehot[0])

      unit_alliance = unit.alliance
      unit_alliance_onehot = np.identity(5)[unit_alliance:unit_alliance+1]
      alliance.append(unit.alliance)

      #print("min(unit.health, 1500): " + str(min(unit.health, 1500)))
      unit_health = int(math.sqrt(min(unit.health, 1500)))
      #print("unit_health: " + str(unit_health))
      unit_health_onehot = np.identity(39)[unit_health:unit_health+1]
      #print("unit_health_onehot: " + str(unit_health_onehot))
      current_health.append(unit_health_onehot[0])

      unit_shield = int(math.sqrt(min(unit.shield, 1000)))
      unit_shield_onehot = np.identity(31)[unit_shield:unit_shield+1]
      current_shields.append(unit_shield_onehot[0])

      unit_energy = int(math.sqrt(min(unit.energy, 200)))
      unit_energy_onehot = np.identity(31)[unit_energy:unit_energy+1]
      current_energy.append(unit_energy_onehot[0])

      cargo_space_used.append(unit.cargo_space_taken)
      cargo_space_maximum.append(unit.cargo_space_max)

      build_progress.append(unit.build_progress)

      current_health_ratio.append(unit.health_ratio)
      current_shield_ratio.append(unit.shield_ratio)
      current_energy_ratio.append(unit.energy_ratio)

      display_type.append(unit.display_type)

      #print("unit.x: " + str(unit.x))
      #print("unit.y: " + str(unit.y))

      x_position.append(bin_array(abs(unit.x), 10))
      y_position.append(bin_array(abs(unit.y), 10))

      is_cloaked.append(unit.cloak)
      is_powered.append(unit.is_powered)
      is_hallucination.append(unit.hallucination)
      is_active.append(unit.active)
      is_on_screen.append(unit.is_in_cargo)
      is_in_cargo.append(unit.is_powered)

      current_minerals.append(unit.mineral_contents)
      current_vespene.append(unit.vespene_contents)

      assigned_harvesters_onehot = np.identity(24)[unit.assigned_harvesters:unit.assigned_harvesters+1]
      ideal_harvesters_onehot = np.identity(17)[unit.ideal_harvesters:unit.ideal_harvesters+1]
      assigned_harvesters.append(assigned_harvesters_onehot[0])
      ideal_harvesters.append(ideal_harvesters_onehot[0])

      weapon_cooldown_onehot = np.identity(32)[unit.weapon_cooldown:unit.weapon_cooldown+1]
      weapon_cooldown.append(weapon_cooldown_onehot[0])

      order_queue_length.append(unit.order_length)
      order_1.append(unit.order_id_0)
      order_2.append(unit.order_id_1)
      order_3.append(unit.order_id_2)
      order_4.append(unit.order_id_3)

      buffs.append([unit.buff_id_0, unit.buff_id_1])

      addon_type.append(unit.addon_unit_type)

      order_progress_1.append(unit.order_progress_0)
      order_progress_2.append(unit.order_progress_1)

      weapon_upgrades_onehot = np.identity(4)[unit.attack_upgrade_level:unit.attack_upgrade_level+1]
      armor_upgrades_onehot = np.identity(4)[unit.armor_upgrade_level:unit.armor_upgrade_level+1]
      shield_upgrades_onehot = np.identity(4)[unit.shield_upgrade_level:unit.shield_upgrade_level+1]

      weapon_upgrades.append(weapon_upgrades_onehot[0])
      armor_upgrades.append(armor_upgrades_onehot[0])
      shield_upgrades.append(shield_upgrades_onehot[0])

      is_selected_onehot = np.identity(2)[unit.is_selected:unit.is_selected+1]
      is_selected.append(is_selected_onehot[0])
    
    '''
    print("unit_type[0].shape: " + str(unit_type[0].shape))
    print("current_health[0].shape: " + str(current_health[0].shape))
    print("current_shields[0].shape: " + str(current_shields[0].shape))
    print("current_energy[0].shape: " + str(current_energy[0].shape))
    print("x_position[0].shape: " + str(x_position[0].shape))
    print("y_position[0].shape: " + str(y_position[0].shape))
    print("assigned_harvesters[0].shape: " + str(assigned_harvesters[0].shape))
    print("ideal_harvesters[0].shape: " + str(ideal_harvesters[0].shape))
    print("weapon_cooldown[0].shape: " + str(weapon_cooldown[0].shape))
    print("weapon_upgrades[0].shape: " + str(weapon_upgrades[0].shape))
    print("armor_upgrades[0].shape: " + str(armor_upgrades[0].shape))
    print("shield_upgrades[0].shape: " + str(shield_upgrades[0].shape))
    print("is_selected[0].shape: " + str(is_selected[0].shape))
    print("")
    '''
    input_list = []
    #print("skip_num: " + str(skip_num))

    length = len(feature_units) - skip_num
    #print("length: " + str(length))
    if length > 100:
      length = 100

    for i in range(0, length):
      entity_array = np.concatenate((unit_type[i], current_health[i], current_shields[i], current_energy[i], x_position[i], y_position[i],
                                          assigned_harvesters[i], ideal_harvesters[i], weapon_cooldown[i], weapon_upgrades[i], armor_upgrades[i],
                                          shield_upgrades[i], is_selected[i]), axis=0, out=None)
      #print("entity_array: " + str(entity_array))

      input_list.append(entity_array)

    if length < 512:
      for i in range(length, 512):
        input_list.append(np.zeros(464))
 
    #print("len(input_list): " + str(len(input_list)))
    input_array = np.array(input_list)
    #print("input_array.shape: " + str(input_array.shape))
    #print("input_array: " + str(input_array))
    #print("")
    
    return input_array


 # feature_player: [ 2 95  0 12 15  0 12  0  0  0  0]
# player_id, minerals, vespene, food_used, food_cap, food_army, food_workers, idle_worker_count, army_count, warp_gate_count, larva_count 
def get_agent_statistics(score_by_category):
  score_by_category = score_by_category.flatten() / 1000.0
  agent_statistics = np.log(score_by_category + 1)

  return agent_statistics


def get_upgrade_obs(feature_units):
    for unit in feature_units:
      #print("unit: " + str(unit))

      unit_info = str(units.get_unit_type(unit.unit_type))
      unit_info = unit_info.split(".")
      unit_race = unit_info[0]
      unit_name = unit_info[1]
      #print("unit_name: " + str(unit_name))

      #print("unit_race: " + str(unit_race))
      #print("unit_name: " + str(unit_name))
      #print("units_new.get_unit_type(unit_race, unit_name): " + str(units_new.get_unit_type(unit_race, unit_name)))
      if unit_race != "Terran" or unit_race != "Neutral":
        #skip_num += 1
        continue

      #print("units_new.get_unit_type(unit_race, unit_name): " + str(units_new.get_unit_type(unit_race, unit_name)))
      unit_category = units_new.get_unit_type(unit_race, unit_name)[1]
      home_upgrade_array = np.zeros(89)
      away_upgrade_array = np.zeros(89)
      #print("unit_category: " + str(unit_category))
      #print("unit.alliance: " + str(unit.alliance))
      if unit.alliance == 1:
        upgrade_name_list = []
        if unit_category == 'Building':
          if unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranStructureArmor')
        elif unit_category == 'Infantry':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranInfantryWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranInfantryWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranInfantryWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranInfantryArmorsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranInfantryArmorsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranInfantryArmorsLevel3')
        elif unit_category == 'Vehicle':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel3')
        elif unit_category == 'Ship':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranShipWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranShipWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranShipWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel3')

        if len(upgrade_name_list) != 0:
          for upgrade_name in upgrade_name_list:
            upgrade_type = int(upgrades_new.get_upgrade_type(upgrade_name))
            home_upgrade_array[upgrade_type] = 1

          return home_upgrade_array, away_upgrade_array
      elif unit.alliance == 4:
        #print("unit.attack_upgrade_level: " + str(unit.attack_upgrade_level))
        #print("unit.armor_upgrade_level: " + str(unit.armor_upgrade_level))
        #print("unit.shield_upgrade_level: " + str(unit.shield_upgrade_level))
        upgrade_name_list = []
        if unit_category == 'Building':
          if unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranStructureArmor')
        elif unit_category == 'Infantry':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranInfantryWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranInfantryWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranInfantryWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranInfantryWeaponsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranInfantryWeaponsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranInfantryWeaponsLevel3')
        elif unit_category == 'Vehicle':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel3')
        elif unit_category == 'Ship':
          if unit.attack_upgrade_level == 1:
            upgrade_name_list.append('TerranShipWeaponsLevel1')
          elif unit.attack_upgrade_level == 2:
            upgrade_name_list.append('TerranShipWeaponsLevel2')
          elif unit.attack_upgrade_level == 3:
            upgrade_name_list.append('TerranShipWeaponsLevel3')

          if unit.armor_upgrade_level == 1:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel1')
          elif unit.armor_upgrade_level == 2:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel2')
          elif unit.armor_upgrade_level == 3:
            upgrade_name_list.append('TerranVehicleAndShipArmorsLevel3')
      
        if len(upgrade_name_list) != 0:
          for upgrade_name in upgrade_name_list:
            upgrade_type = int(upgrades_new.get_upgrade_type(upgrade_name))
            away_upgrade_array[upgrade_type] = 1

          return home_upgrade_array, away_upgrade_array

      return -1


race_list = ["Protoss", "Terran", "Zerg", "Unknown"]
def get_race_onehot(home_race, away_race):
  home_race_index = race_list.index(home_race)
  away_race_index = race_list.index(away_race)
  home_race_onehot = np.identity(5)[home_race_index:home_race_index+1]
  away_race_onehot = np.identity(5)[away_race_index:away_race_index+1]

  return np.array([home_race_onehot[0], away_race_onehot[0]]).flatten()


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

with tf.device('/cpu:0'):
  pos_encoding = positional_encoding(16000, 64)


def get_gameloop_obs(game_loop):
  time = pos_encoding[:, game_loop[0], :]
  #print("time.shape : " + str(time.shape))

  return time.numpy().flatten()


def get_spatial_obs(feature_screen):
  spatial_input  = np.reshape(feature_screen, [1,128,128,27])

  return spatial_input