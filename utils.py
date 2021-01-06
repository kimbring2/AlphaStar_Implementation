from pysc2.lib import actions, features, units
import numpy as np
import units_new
import upgrades_new
import math

import os
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

################# Action part #################
_NO_OP = [actions.FUNCTIONS.no_op]
_MOVE_SCREEN = [actions.FUNCTIONS.Move_screen]
_MOVE_CAMERA = [actions.FUNCTIONS.move_camera]
_SELECT_ARMY = [actions.FUNCTIONS.select_army]

# public action
_SELECT_POINT_SELECT = [actions.FUNCTIONS.select_point, actions.SelectPointAct.select]
_SELECT_POINT_TOGGLE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.toggle]
_SELECT_POINT_SELECT_ALL_TYPE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.select_all_type]
_SELECT_POINT_ADD_ALL_TYPE = [actions.FUNCTIONS.select_point, actions.SelectPointAct.add_all_type]

_SELECT_RECT_SELECT = [actions.FUNCTIONS.select_rect, actions.SelectAdd.select]
_SELECT_RECT_ADD = [actions.FUNCTIONS.select_rect, actions.SelectAdd.add]

_SELECT_CONTROL_GROUP_RECALL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.recall]
_SELECT_CONTROL_GROUP_SET = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.set]
_SELECT_CONTROL_GROUP_APPEND = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.append]
_SELECT_CONTROL_GROUP_SET_AND_STEAL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.set_and_steal]
_SELECT_CONTROL_GROUP_APPEND_AND_STEAL = [actions.FUNCTIONS.select_control_group, actions.ControlGroupAct.append_and_steal]

_SELECT_UNIT_SELECT = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.select]
_SELECT_UNIT_DESELECT = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.deselect]
_SELECT_UNIT_SELECT_ALL_TYPE = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.select_all_type]
_SELECT_UNIT_DESELECT_ALL_TYPE = [actions.FUNCTIONS.select_unit, actions.SelectUnitAct.deselect_all_type]

_SELECT_IDLE_WORKER_SELECT = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.select]
_SELECT_IDLE_WORKER_ADD = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.add]
_SELECT_IDLE_WORKER_SELECT_ALL = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.select_all]
_SELECT_IDLE_WORKER_ADD_ALL = [actions.FUNCTIONS.select_idle_worker, actions.SelectWorker.add_all]

_SMART_SCREEN = [actions.FUNCTIONS.Smart_screen]
_SMART_MINIMAP = [actions.FUNCTIONS.Smart_minimap]

_ATTACK_SCREEN = [actions.FUNCTIONS.Attack_screen]
_ATTACK_MINIMAP = [actions.FUNCTIONS.Attack_minimap]

_STOP_QUICK = [actions.FUNCTIONS.Stop_quick]
_CANCEL_QUICK = [actions.FUNCTIONS.Cancel_quick]
_LIFT_QUICK = [actions.FUNCTIONS.Lift_quick]
_LAND_SCREEN = [actions.FUNCTIONS.Land_screen]
_CANCEL_LAST_QUICK = [actions.FUNCTIONS.Cancel_Last_quick]
_RALLY_WORKERS_SCREEN = [actions.FUNCTIONS.Rally_Workers_screen]
_PATROL_SCREEN = [actions.FUNCTIONS.Patrol_screen]
_BUILD_QUEUE = [actions.FUNCTIONS.build_queue]
_HALT_QUICK = [actions.FUNCTIONS.Halt_quick]
_HOLDPOSITION_QUICK = [actions.FUNCTIONS.HoldPosition_quick]

_RALLY_UNITS_SCREEN = [actions.FUNCTIONS.Rally_Units_screen]
_RALLY_UNITS_MINIMAP = [actions.FUNCTIONS.Rally_Units_minimap]
_RALLY_BUILDING_SCREEN = [actions.FUNCTIONS.Rally_Building_screen]
_RALLY_BUILDING_MINIMAP = [actions.FUNCTIONS.Rally_Building_minimap]
_RALLY_WORKERS_SCREEN = [actions.FUNCTIONS.Rally_Workers_screen]
_RALLY_WORKERS_MINIMAP = [actions.FUNCTIONS.Rally_Workers_minimap]

_UNLOAD = [actions.FUNCTIONS.unload]
_LOADALL_QUICK = [actions.FUNCTIONS.LoadAll_quick]
_UNLOADALL_QUICK = [actions.FUNCTIONS.UnloadAll_quick]

# terran action
_RETURN_SCV_QUICK = [actions.FUNCTIONS.Harvest_Return_SCV_quick]

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
_BUILD_GHOSTACADEMY_SCREEN = [actions.FUNCTIONS.Build_GhostAcademy_screen]
_BUILD_MISSILETURRET_SCREEN = [actions.FUNCTIONS.Build_MissileTurret_screen]

_TRAIN_MARINE_QUICK = [actions.FUNCTIONS.Train_Marine_quick]
_TRAIN_MARAUDER_QUICK = [actions.FUNCTIONS.Train_Marauder_quick]
_TRAIN_SCV_QUICK = [actions.FUNCTIONS.Train_SCV_quick]
_TRAIN_SIEGETANK_QUICK = [actions.FUNCTIONS.Train_SiegeTank_quick]
_TRAIN_MEDIVAC_QUICK = [actions.FUNCTIONS.Train_Medivac_quick]
_TRAIN_REAPER_QUICK = [actions.FUNCTIONS.Train_Reaper_quick]
_TRAIN_HELLION_QUICK = [actions.FUNCTIONS.Train_Hellion_quick]
_TRAIN_VIKINGFIGHTER_QUICK = [actions.FUNCTIONS.Train_VikingFighter_quick]
_TRAIN_RIBERATOR_QUICK = [actions.FUNCTIONS.Train_Liberator_quick]
_TRAIN_WIDOWMINE_QUICK = [actions.FUNCTIONS.Train_WidowMine_quick]
_TRAIN_RAVEN_QUICK = [actions.FUNCTIONS.Train_Raven_quick]
_TRAIN_BANSHEE_QUICK = [actions.FUNCTIONS.Train_Banshee_quick]
_TRAIN_CYCLONE_QUICK = [actions.FUNCTIONS.Train_Cyclone_quick]

_HARVEST_GATHER_SCREEN = [actions.FUNCTIONS.Harvest_Gather_screen]
_HARVEST_GATHER_SCV_SCREEN = [actions.FUNCTIONS.Harvest_Gather_SCV_screen]
_HARVEST_RETURN_QUICK = [actions.FUNCTIONS.Harvest_Return_quick]

_MORPH_SIEGEMODE_QUICK  = [actions.FUNCTIONS.Morph_SiegeMode_quick]
_MORPH_SUPPLYDEPOT_LOWER_QUICK = [actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick]
_MORPH_SUPPLYDEPOT_RAISE_QUICK = [actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick]
_MORPH_ORBITALCOMMAND_QUICK = [actions.FUNCTIONS.Morph_OrbitalCommand_quick]

_EFFECT_COOLDOWNMULE_SCREEN = [actions.FUNCTIONS.Effect_CalldownMULE_screen]
_EFFECT_KD8CHARGE_SCREEN = [actions.FUNCTIONS.Effect_KD8Charge_screen]
_EFFECT_SPRAY_SCREEN = [actions.FUNCTIONS.Effect_Spray_screen]
_EFFECT_MEDIVACIGNITEAFTERBURNERS_QUICK = [actions.FUNCTIONS.Effect_MedivacIgniteAfterburners_quick]
_EFFECT_STIM_QUICK = [actions.FUNCTIONS.Effect_Stim_quick]
_EFFECT_STIM_MARAUDER_QUICK = [actions.FUNCTIONS.Effect_Stim_Marauder_quick]
_EFFECT_STIM_MARAUDER_REDIRECT_QUICK = [actions.FUNCTIONS.Effect_Stim_Marauder_Redirect_quick]
_EFFECT_STIM_MARINE_QUICK = [actions.FUNCTIONS.Effect_Stim_Marine_quick]
_EFFECT_STIM_MARINE_REDIRECT_QUICK = [actions.FUNCTIONS.Effect_Stim_Marine_Redirect_quick]
_EFFECT_SUPPLYDROP_SCREEN = [actions.FUNCTIONS.Effect_SupplyDrop_screen]
_EFFECT_TACTICALJUMP_SCREEN = [actions.FUNCTIONS.Effect_TacticalJump_screen]
_EFFECT_TACTICALJUMP_MINIMAP = [actions.FUNCTIONS.Effect_TacticalJump_minimap]
_EFFECT_WIDOWMINEATTACK_SCREEN = [actions.FUNCTIONS.Effect_WidowMineAttack_screen]
_EFFECT_WIDOWMINEATTACK_AUTOCAST = [actions.FUNCTIONS.Effect_WidowMineAttack_autocast]
_EFFECT_YAMATOGUN_SCREEN = [actions.FUNCTIONS.Effect_YamatoGun_screen]
_EFFECT_REPAIR_SCREEN = [actions.FUNCTIONS.Effect_Repair_screen]
_EFFECT_REPAIR_AUTOCAST = [actions.FUNCTIONS.Effect_Repair_autocast]
_EFFECT_REPAIR_MULE_SCREEN = [actions.FUNCTIONS.Effect_Repair_Mule_screen]
_EFFECT_REPAIR_MULE_AUTOCAST = [actions.FUNCTIONS.Effect_Repair_Mule_autocast]
_EFFECT_REPAIR_REPAIRDRONE_SCREEN = [actions.FUNCTIONS.Effect_Repair_RepairDrone_screen]
_EFFECT_REPAIR_REPAIRDRONE_AUTOCAST = [actions.FUNCTIONS.Effect_Repair_RepairDrone_autocast]
_EFFECT_REPAIR_SCV_SCREEN = [actions.FUNCTIONS.Effect_Repair_SCV_screen]
_EFFECT_REPAIR_SCV_AUTOCAST = [actions.FUNCTIONS.Effect_Repair_SCV_autocast]
_EFFECT_SCAN_SCREEN = [actions.FUNCTIONS.Effect_Scan_screen]
_EFFECT_SCAN_MINIMAP = [actions.FUNCTIONS.Effect_Scan_minimap]

_RESEARCH_STIMPACK_QUICK = [actions.FUNCTIONS.Research_Stimpack_quick]
_RESEARCH_COMBATSHIELD_QUICK = [actions.FUNCTIONS.Research_CombatShield_quick]
_RESEARCH_TERRANINFANTRYARMOR_QUICK = [actions.FUNCTIONS.Research_TerranInfantryArmor_quick]
_RESEARCH_TERRANINFANTRYWEAPONS_QUICK = [actions.FUNCTIONS.Research_TerranInfantryWeapons_quick]
_RESEARCH_TERRANSHIPWEAPONS_QUICK = [actions.FUNCTIONS.Research_TerranShipWeapons_quick]
_RESEARCH_TERRANSTRUCTUREARMORUPGRADE_QUICK = [actions.FUNCTIONS.Research_TerranStructureArmorUpgrade_quick]
_RESEARCH_TERRANVEHICLEANDSHIPPLATING_QUICK = [actions.FUNCTIONS.Research_TerranVehicleAndShipPlating_quick]
_RESEARCH_TERRANVEHICLEWEAPONS_QUICK = [actions.FUNCTIONS.Research_TerranVehicleWeapons_quick]

_RALLY_COMMANDCENTER_SCREEN = [actions.FUNCTIONS.Rally_CommandCenter_screen]
_RALLY_COMMANDCENTER_MINIMAP = [actions.FUNCTIONS.Rally_CommandCenter_minimap]

_UNLOADALL_BUNKER_QUICK = [actions.FUNCTIONS.UnloadAll_Bunker_quick]
_UNLOADALL_COMMANDCENTER_QUICK = [actions.FUNCTIONS.UnloadAll_CommandCenter_quick]
_UNLOADALLAT_SCREEN = [actions.FUNCTIONS.UnloadAllAt_screen]
_UNLOADALLAT_MINIMAP = [actions.FUNCTIONS.UnloadAllAt_minimap]
_UNLOADALLAT_MEDIVAC_SCREEN = [actions.FUNCTIONS.UnloadAllAt_Medivac_screen]
_UNLOADALLAT_MEDIVAC_MINIMAP = [actions.FUNCTIONS.UnloadAllAt_Medivac_minimap]

action_type_list = [_NO_OP, _BUILD_SUPPLYDEPOT_SCREEN, _BUILD_BARRACKS_SCREEN, _BUILD_REFINERY_SCREEN, _BUILD_TECHLAB_SCREEN, _BUILD_COMMANDCENTER_SCREEN, 
                      _BUILD_REACTOR_QUICK, _BUILD_BUNKER_SCREEN, _BUILD_STARPORT_SCREEN, _BUILD_FACTORY_SCREEN, _HALT_QUICK, _RESEARCH_COMBATSHIELD_QUICK,
                      _TRAIN_MARINE_QUICK, _TRAIN_MARAUDER_QUICK, _TRAIN_SCV_QUICK, _TRAIN_SIEGETANK_QUICK, _TRAIN_MEDIVAC_QUICK, _SELECT_UNIT_DESELECT, 
                      _SELECT_UNIT_SELECT_ALL_TYPE, _SELECT_UNIT_DESELECT_ALL_TYPE, _TRAIN_REAPER_QUICK, _TRAIN_RIBERATOR_QUICK, _TRAIN_WIDOWMINE_QUICK,
                      _RETURN_SCV_QUICK, _HARVEST_GATHER_SCREEN, _HARVEST_GATHER_SCV_SCREEN, _PATROL_SCREEN, _SELECT_UNIT_SELECT, _HOLDPOSITION_QUICK,
                      _LIFT_QUICK, _MORPH_SUPPLYDEPOT_LOWER_QUICK, _LAND_SCREEN, _BUILD_TECHLAB_QUICK, _EFFECT_MEDIVACIGNITEAFTERBURNERS_QUICK,
                      _RESEARCH_STIMPACK_QUICK, _SELECT_POINT_SELECT, _SELECT_POINT_TOGGLE, _SELECT_POINT_SELECT_ALL_TYPE, _SELECT_POINT_ADD_ALL_TYPE,
                      _ATTACK_SCREEN, _ATTACK_MINIMAP, _SMART_SCREEN, _SMART_MINIMAP, _MORPH_ORBITALCOMMAND_QUICK, _BUILD_ENGINNERINGBAY_SCREEN,
                      _SELECT_RECT_SELECT, _SELECT_RECT_ADD, _SELECT_IDLE_WORKER_SELECT, _SELECT_IDLE_WORKER_SELECT_ALL, _SELECT_IDLE_WORKER_ADD_ALL, 
                      _SELECT_IDLE_WORKER_ADD, _SELECT_CONTROL_GROUP_RECALL, _SELECT_CONTROL_GROUP_SET, _SELECT_CONTROL_GROUP_APPEND, _SELECT_CONTROL_GROUP_SET_AND_STEAL,
                      _SELECT_CONTROL_GROUP_APPEND_AND_STEAL, _SELECT_ARMY, _BUILD_ARMORY_SCREEN, _BUILD_REACTOR_SCREEN, _STOP_QUICK, _UNLOADALL_QUICK,
                      _MOVE_SCREEN, _MOVE_CAMERA, _CANCEL_LAST_QUICK, _RALLY_WORKERS_SCREEN, _HARVEST_RETURN_QUICK, _TRAIN_HELLION_QUICK, _CANCEL_QUICK,
                      _EFFECT_COOLDOWNMULE_SCREEN, _MORPH_SUPPLYDEPOT_RAISE_QUICK, _BUILD_QUEUE, _EFFECT_KD8CHARGE_SCREEN, _UNLOAD, _EFFECT_SPRAY_SCREEN,
                      _TRAIN_VIKINGFIGHTER_QUICK, _SELECT_POINT_SELECT, _SELECT_POINT_TOGGLE, _SELECT_POINT_SELECT_ALL_TYPE, _SELECT_POINT_ADD_ALL_TYPE,
                      _MORPH_SIEGEMODE_QUICK, _RESEARCH_TERRANINFANTRYARMOR_QUICK, _RESEARCH_TERRANINFANTRYWEAPONS_QUICK, _RESEARCH_TERRANSHIPWEAPONS_QUICK,
                      _RESEARCH_TERRANSTRUCTUREARMORUPGRADE_QUICK, _RESEARCH_TERRANVEHICLEANDSHIPPLATING_QUICK, _RESEARCH_TERRANVEHICLEWEAPONS_QUICK,
                      _EFFECT_STIM_QUICK, _EFFECT_STIM_MARAUDER_QUICK, _EFFECT_STIM_MARAUDER_REDIRECT_QUICK, _EFFECT_STIM_MARINE_QUICK, _EFFECT_STIM_MARINE_REDIRECT_QUICK,
                      _TRAIN_RAVEN_QUICK, _BUILD_GHOSTACADEMY_SCREEN, _EFFECT_SUPPLYDROP_SCREEN, _EFFECT_TACTICALJUMP_SCREEN, _EFFECT_TACTICALJUMP_MINIMAP,
                      _EFFECT_WIDOWMINEATTACK_SCREEN, _EFFECT_WIDOWMINEATTACK_AUTOCAST, _EFFECT_YAMATOGUN_SCREEN, _EFFECT_REPAIR_SCREEN, _EFFECT_REPAIR_AUTOCAST,
                      _EFFECT_REPAIR_MULE_SCREEN, _EFFECT_REPAIR_MULE_AUTOCAST, _EFFECT_REPAIR_REPAIRDRONE_SCREEN, _EFFECT_REPAIR_REPAIRDRONE_AUTOCAST, 
                      _EFFECT_REPAIR_SCV_SCREEN, _EFFECT_REPAIR_SCV_AUTOCAST, _TRAIN_BANSHEE_QUICK, _BUILD_MISSILETURRET_SCREEN, _UNLOADALL_BUNKER_QUICK,
                      _UNLOADALL_COMMANDCENTER_QUICK, _UNLOADALLAT_SCREEN, _UNLOADALLAT_MINIMAP, _UNLOADALLAT_MEDIVAC_SCREEN, _UNLOADALLAT_MEDIVAC_MINIMAP,
                      _RALLY_UNITS_SCREEN, _RALLY_UNITS_MINIMAP, _RALLY_BUILDING_SCREEN, _RALLY_BUILDING_MINIMAP, _RALLY_WORKERS_SCREEN, _RALLY_WORKERS_MINIMAP,
                      _RALLY_COMMANDCENTER_SCREEN, _RALLY_COMMANDCENTER_MINIMAP, _TRAIN_CYCLONE_QUICK]

'''
action_type_list = []
for action in actions.FUNCTIONS:
  #print("action.id: " + str(action.id))
  #print("int(action.id): " + str(int(action.id)))
  if int(action.id) == 2:
    action_type_list.append([action, actions.SelectPointAct.select])
    action_type_list.append([action, actions.SelectPointAct.toggle])
    action_type_list.append([action, actions.SelectPointAct.select_all_type])
    action_type_list.append([action, actions.SelectPointAct.add_all_type])
  elif int(action.id) == 3:
    action_type_list.append([action, actions.SelectAdd.select])
    action_type_list.append([action, actions.SelectAdd.add])
  elif int(action.id) == 4:
    action_type_list.append([action, actions.ControlGroupAct.recall])
    action_type_list.append([action, actions.ControlGroupAct.set])
    action_type_list.append([action, actions.ControlGroupAct.append])
    action_type_list.append([action, actions.ControlGroupAct.set_and_steal])
    action_type_list.append([action, actions.ControlGroupAct.append_and_steal])
  elif int(action.id) == 5:
    action_type_list.append([action, actions.SelectUnitAct.select])
    action_type_list.append([action, actions.SelectUnitAct.deselect])
    action_type_list.append([action, actions.SelectUnitAct.select_all_type])
    action_type_list.append([action, actions.SelectUnitAct.deselect_all_type])
  elif int(action.id) == 6:   
    action_type_list.append([action, actions.SelectWorker.select])
    action_type_list.append([action, actions.SelectWorker.add])
    action_type_list.append([action, actions.SelectWorker.select_all])
    action_type_list.append([action, actions.SelectWorker.add_all])
  else:
    action_type_list.append([action])
'''
action_id_list = []
for action_type in action_type_list:
  if len(action_type) != 1:
    action_id_list.append([int(action_type[0].id), int(action_type[1])])
  else:
    action_id_list.append([int(action_type[0].id)])

action_len = len(action_type_list)
#print("action_len: " + str(action_len))

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

  #print("game_loop: " + str(game_loop))
  time = get_gameloop_obs(game_loop)
  #print("time.shape: " + str(time.shape))

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


def get_action_from_prediction(agent, observation, action_type_index, selected_units, target_unit, 
                                   screen_target_location1_x, screen_target_location1_y, screen_target_location2_x, screen_target_location2_y,
                                   minimap_target_location_x, minimap_target_location_y):
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

  argument = []
  #print("action_type.args: " + str(action_type.args))
  for action_type_arg in action_type_name.args:    
    if action_type_arg.id == 0:
      # action_type_arg.name: screen
      # action_type_arg.sizes: (0, 0)
      argument.append([screen_target_location1_x[0], screen_target_location1_y[0]])
    elif action_type_arg.id == 1:
      # action_type_arg.name: minimap
      # action_type_arg.sizes: (0, 0)
      argument.append([minimap_target_location_x[0], minimap_target_location_y[0]])
    elif action_type_arg.id == 2:
      # action_type_arg.name: screen2
      # action_type_arg.sizes: (0, 0)
      argument.append([screen_target_location2_x[0], screen_target_location2_y[0]])
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


def get_supervised_loss(batch_size, loss_function, predict_value, trajectorys):
  scce = loss_function
  all_losses = 0 
  for i in range(0, batch_size):
      action_type_pred = predict_value[1][i]
      selected_units_pred = predict_value[2][i]
      target_unit_pred = predict_value[4][i]
      screen_target_location1_pred = predict_value[6][i]
      screen_target_location2_pred = predict_value[8][i]
      minimap_target_location_pred = predict_value[10][i]

      acts_types_agent = action_type_list[action_type_pred]
      #print("acts_types_agent:" + str(acts_types_agent))
      #if len(acts_types_agent) == 2:
        #print("acts_types_agent[1]: " + str(acts_types_agent[1]))

      acts_human = trajectorys[i][1]
      #print("acts_human:" + str(acts_human))
      #print("")
      #print("")
      #print("")

      all_losses = 0 
      for act_human in acts_human:
        #print("act_human:" + str(act_human))
        human_function = act_human.function
        human_argument = act_human.arguments

        # select_point, select_rect, select_control_group, select_unit, select_idle_worker
        if int(human_function) == 2 or int(human_function) == 3 or int(human_function) == 4 or int(human_function) == 5 \
            or int(human_function) == 6:
          human_action_with_argument = [int(human_function), int(human_argument[0][0])]
          human_action_index = action_id_list.index(human_action_with_argument)
        else:
          human_action_name = str(human_function).split('.')[-1]
          human_action_index = action_id_list.index([int(actions._Functions[human_action_name])])

        if human_action_index != predict_value[1][i].numpy():
          action_true = [human_action_index]
          action_pred = predict_value[0][i]

          #print("action_len: " + str(action_len))
          #print("action_true: " + str(action_true))
          #print("tf.argmax(action_pred): " + str(tf.argmax(action_pred)))
          action_loss = scce(action_true, action_pred)
          #print("action_loss: " + str(action_loss))
          #print("")
          #print("")
          #print("")

          all_losses += action_loss
        else:
          arg_list = action_type_list[human_action_index][0].args
          for j, arg in enumerate(arg_list):
            human_arg = human_argument[j]
            
            if arg.id == 0:
              # action_type_arg.name: screen
              # action_type_arg.sizes: (0, 0)human_arg[0] * 256 + human_arg[1]
              screen_position1_human = [int(human_arg[0]) * 256 + int(human_arg[1])]
              screen_position1_agent = [screen_target_location1_pred]
              screen1_loss = scce(screen_position1_human, screen_position1_agent)
              all_losses += 0.1 * screen1_loss
            elif arg.id == 1:
              # action_type_arg.name: minimap
              # action_type_arg.sizes: (0, 0)
              minimap_position_human = [int(human_arg[0]) * 128 + int(human_arg[1])]
              minimap_position_agent = [minimap_target_location_pred]
              minimap_loss = scce(minimap_position_human, minimap_position_agent)
              all_losses += 0.1 * minimap_loss
            elif arg.id == 2:
              # action_type_arg.name: screen2
              # action_type_arg.sizes: (0, 0)
              screen_position2_human = [int(human_arg[0]) * 256 + int(human_arg[1])]
              screen_position2_agent = [screen_target_location2_pred]
              screen2_loss = scce(screen_position2_human, screen_position2_agent)
              all_losses += 0.1 * screen2_loss
            elif arg.id == 3:
              pass
              # action_type_arg.name: queued
              # action_type_arg.sizes: (2,)
              #act_name = 'now'
              #if act_name == 'now':
              #  argument.append([0])
              #elif act_name == 'queued':
              #  argument.append([1])
            elif arg.id == 4:
              # action_type_arg.name: control_group_act
              # action_type_arg.sizes: (5,)
              control_group_act_human = [int(human_arg[0])]
              #print("control_group_act_human: " + str(control_group_act_human))
              #print("int(acts_types_agent[1]).numpy(): " + str(int(acts_types_agent[1]).numpy()))
              control_group_act_loss = tf.keras.losses.MeanSquaredError()(control_group_act_human, [int(acts_types_agent[1])]).numpy()
              #print("control_group_act_loss: " + str(control_group_act_loss))
              #print("all_losses: " + str(all_losses))
              all_losses += 0.1 * control_group_act_loss
            elif arg.id == 5:
              # action_type_arg.name: control_group_id
              # action_type_arg.sizes: (10,)
              control_group_id_human = [int(human_arg[0])]
              control_group_id_agent = [selected_units_pred]
              control_group_id_loss = scce(control_group_id_human, control_group_id_agent)
              all_losses += 0.1 * control_group_id_loss
            elif arg.id == 6:
              # action_type_arg.name: select_point_act
              # action_type_arg.sizes: (4,)
              select_point_act_human = [int(human_arg[0])]
              #print("select_point_act_human: " + str(select_point_act_human))
              #print("int(acts_types_agent[1]): " + str(int(acts_types_agent[1])))
              select_point_act_loss = tf.keras.losses.MeanSquaredError()(select_point_act_human, [int(acts_types_agent[1])]).numpy()
              #print("select_point_act_loss: " + str(select_point_act_loss))
              #print("all_losses: " + str(all_losses))
              all_losses += 0.1 * select_point_act_loss
            elif arg.id == 7:
              # action_type_arg.name: select_add
              # action_type_arg.sizes: (2,)
              select_add_human = [int(human_arg[0])]
              #print("select_add_human: " + str(select_add_human))
              #print("int(acts_types_agent[1]): " + str(int(acts_types_agent[1])))
              select_add_loss = tf.keras.losses.MeanSquaredError()(select_add_human, [int(acts_types_agent[1])]).numpy()
              #print("select_add_loss: " + str(select_add_loss))
              #print("all_losses: " + str(all_losses))
              all_losses += 0.1 * select_add_loss
            elif arg.id == 8:
              # action_type_arg.name: select_unit_act
              # action_type_arg.sizes: (4,)
              select_unit_act_human = [int(human_arg[0])]
              #print("select_unit_act_human: " + str(select_unit_act_human))
              #print("int(acts_types_agent[1]): " + str(int(acts_types_agent[1])))
              select_unit_act_loss = tf.keras.losses.MeanSquaredError()(select_unit_act_human, [int(acts_types_agent[1])]).numpy()
              #print("select_unit_act_loss: " + str(select_unit_act_loss))
              #print("all_losses: " + str(all_losses))
              all_losses += 0.1 * select_unit_act_loss
            elif arg.id == 9:
              # action_type_arg.name: select_unit_id
              # action_type_arg.sizes: (500,)
              selected_unit_id_human = [int(human_arg[0])]
              selected_unit_id_agent = [selected_units_pred]
              selected_unit_id_loss = scce(selected_unit_id_human, selected_unit_id_agent)
              all_losses += 0.1 * selected_unit_id_loss
            elif arg.id == 10:
              # action_type_arg.name: select_worker
              # action_type_arg.sizes: (4,)
              select_worker_human = [int(human_arg[0])]
              #print("select_worker_human: " + str(select_worker_human))
              #print("int(acts_types_agent[1]): " + str(int(acts_types_agent[1])))
              select_worker_loss = tf.keras.losses.MeanSquaredError()(select_worker_human, [int(acts_types_agent[1])]).numpy()
              #print("select_worker_loss: " + str(select_worker_loss))
              #print("all_losses: " + str(all_losses))
              all_losses += 0.1 * select_worker_loss
            elif arg.id == 11:
              # action_type_arg.name: build_queue_id
              # action_type_arg.sizes: (10,)
              build_queue_id_human = [int(human_arg[0])]
              build_queue_id_agent = [selected_units_pred]
              build_queue_id_loss = scce(build_queue_id_human, build_queue_id_agent)
              all_losses += 0.1 * build_queue_id_loss
            elif arg.id == 12:
              # action_type_arg.name: unload_id
              # action_type_arg.sizes: (500,)
              unload_id_human = [int(human_arg[0])]
              unload_id_agent = [selected_units_pred]
              unload_id_loss = scce(unload_id_human, unload_id_agent)
              all_losses += 0.1 * unload_id_loss
              
      action_true = [0]
      action_pred = predict_value[0][i]
      action_loss = scce(action_true, action_pred)
      all_losses += 0.0001 * action_loss

      selected_unit_id_human = [0]
      selected_unit_id_agent = [selected_units_pred]
      selected_unit_id_loss = scce(selected_unit_id_human, selected_unit_id_agent)
      all_losses += 0.0001 * selected_unit_id_loss

      build_queue_id_human = [0]
      build_queue_id_agent = [selected_units_pred]
      build_queue_id_loss = scce(build_queue_id_human, build_queue_id_agent)
      all_losses += 0.0001 * build_queue_id_loss

      unload_id_human = [0]
      unload_id_agent = [selected_units_pred]
      unload_id_loss = scce(unload_id_human, unload_id_agent)
      all_losses += 0.0001 * unload_id_loss

      screen_position2_human = [0]
      screen_position2_agent = [screen_target_location2_pred]
      screen1_loss = scce(screen_position2_human, screen_position2_agent)
      all_losses += 0.0001 * screen1_loss

      screen_position1_human = [0]
      screen_position1_agent = [screen_target_location1_pred]
      screen2_loss = scce(screen_position1_human, screen_position1_agent)
      all_losses += 0.0001 * screen2_loss

      minimap_position_human = [0]
      minimap_position_agent = [minimap_target_location_pred]
      minimap_loss = scce(minimap_position_human, minimap_position_agent)
      all_losses += 0.0001 * minimap_loss

  return all_losses


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
      #print("unit: " + str(unit))
      try:
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

        unit_alliance = unit.alliance
        unit_alliance_onehot = np.identity(5)[unit_alliance:unit_alliance+1]

        unit_health = int(math.sqrt(min(unit.health, 1500)))
        unit_health_onehot = np.identity(39)[unit_health:unit_health+1]

        unit_shield = int(math.sqrt(min(unit.shield, 1000)))
        unit_shield_onehot = np.identity(31)[unit_shield:unit_shield+1]

        unit_energy = int(math.sqrt(min(unit.energy, 200)))
        unit_energy_onehot = np.identity(31)[unit_energy:unit_energy+1]

        assigned_harvesters_onehot = np.identity(24)[unit.assigned_harvesters:unit.assigned_harvesters+1]
        ideal_harvesters_onehot = np.identity(17)[unit.ideal_harvesters:unit.ideal_harvesters+1]

        weapon_cooldown_onehot = np.identity(32)[unit.weapon_cooldown//8:unit.weapon_cooldown//8 + 1]
        
        weapon_upgrades_onehot = np.identity(4)[unit.attack_upgrade_level:unit.attack_upgrade_level+1]
        armor_upgrades_onehot = np.identity(4)[unit.armor_upgrade_level:unit.armor_upgrade_level+1]
        shield_upgrades_onehot = np.identity(4)[unit.shield_upgrade_level:unit.shield_upgrade_level+1]

        is_selected_onehot = np.identity(2)[unit.is_selected:unit.is_selected+1]
  
        unit_type.append(unit_info_onehot[0])
        alliance.append(unit.alliance)
        current_health.append(unit_health_onehot[0])
        current_shields.append(unit_shield_onehot[0])
        current_energy.append(unit_energy_onehot[0])
        cargo_space_used.append(unit.cargo_space_taken)
        cargo_space_maximum.append(unit.cargo_space_max)
        build_progress.append(unit.build_progress)
        current_health_ratio.append(unit.health_ratio)
        current_shield_ratio.append(unit.shield_ratio)
        current_energy_ratio.append(unit.energy_ratio)
        display_type.append(unit.display_type)
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
        assigned_harvesters.append(assigned_harvesters_onehot[0])
        ideal_harvesters.append(ideal_harvesters_onehot[0])
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
        weapon_upgrades.append(weapon_upgrades_onehot[0])
        armor_upgrades.append(armor_upgrades_onehot[0])
        shield_upgrades.append(shield_upgrades_onehot[0])
        is_selected.append(is_selected_onehot[0])
      except:
        unit_info = str(units.get_unit_type(unit.unit_type))
        unit_info = unit_info.split(".")
        #print("unit_info: " + str(unit_info))
        skip_num += 1
        continue
    
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
  pos_encoding = positional_encoding(32000, 64)


def get_gameloop_obs(game_loop):
  time = pos_encoding[:, game_loop[0], :]
  #print("time.shape : " + str(time.shape))

  return time.numpy().flatten()


def get_spatial_obs(feature_screen):
  spatial_input  = np.reshape(feature_screen, [1,128,128,27])

  return spatial_input