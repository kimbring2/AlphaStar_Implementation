from pysc2.lib import actions, features, units
import numpy as np
import units_new
import upgrades_new
import math
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


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


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
    for unit in feature_units:
      unit_info = str(units.get_unit_type(unit.unit_type))
      unit_info = unit_info.split(".")
      unit_race = unit_info[0]
      unit_name = unit_info[1]

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

      weapon_cooldown_onehot =  np.identity(32)[unit.weapon_cooldown:unit.weapon_cooldown+1]
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
    #print("len(current_health): " + str(len(current_health)))

    length = len(feature_units)
    if length > 100:
      length = 100

    for i in range(0, length):
      entity_array = np.concatenate((unit_type[i], current_health[i], current_shields[i], current_energy[i], x_position[i], y_position[i],
                                          assigned_harvesters[i], ideal_harvesters[i], weapon_cooldown[i], weapon_upgrades[i], armor_upgrades[i],
                                          shield_upgrades[i], is_selected[i]), axis=0, out=None)
      #print("input_array.shape: " + str(input_array.shape))

      input_list.append(entity_array)

    if length < 512:
      for i in range(length, 512):
        input_list.append(np.zeros(464))
 
    #print("len(input_list): " + str(len(input_list)))
    input_array = np.array(input_list)
    #print("input_array.shape: " + str(input_array.shape))
    #print("")
    
    return input_array


 # feature_player: [ 2 95  0 12 15  0 12  0  0  0  0]
# player_id, minerals, vespene, food_used, food_cap, food_army, food_workers, idle_worker_count, army_count, warp_gate_count, larva_count 
def get_agent_statistics(score_by_category):
  score_by_category = score_by_category.flatten()
  agent_statistics = np.log(score_by_category + 1)

  return agent_statistics


def get_upgrade_obs(feature_units):
    for unit in feature_units:
      unit_info = str(units.get_unit_type(unit.unit_type))
      unit_info = unit_info.split(".")
      unit_race = unit_info[0]
      unit_name = unit_info[1]
      #print("unit_name: " + str(unit_name))

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

pos_encoding = positional_encoding(16000, 64)


def get_gameloop_obs(game_loop):
  time = pos_encoding[:, game_loop[0], :]
  #print("time.shape : " + str(time.shape))

  return time.numpy().flatten()