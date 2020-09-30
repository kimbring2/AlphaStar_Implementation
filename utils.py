from pysc2.lib import actions, features, units
import numpy as np
import units_new
import math


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