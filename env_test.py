from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions, features, units
import sys
import units_new
import upgrades_new

from utils import get_entity_obs, get_upgrade_obs, get_gameloop_obs, get_race_onehot, get_agent_statistics
from network import EntityEncoder, SpatialEncoder, Core, ActionTypeHead, SelectedUnitsHead, TargetUnitHead, LocationHead
from trajectory import Trajectory

import random
import time
import math
import statistics
import numpy as np
import tensorflow as tf

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_name = 'Simple128'
#players = [sc2_env.Agent(sc2_env.Race['terran']), 
#           sc2_env.Bot(sc2_env.Race['protoss'], sc2_env.Difficulty.very_easy)]
players = [sc2_env.Agent(sc2_env.Race['terran']), 
            sc2_env.Agent(sc2_env.Race['terran'])]

feature_screen_size = 128
feature_minimap_size = 64
rgb_screen_size = None
rgb_minimap_size = None
action_space = None
use_feature_units = True
step_mul = 8
game_steps_per_episode = None
disable_fog = True
visualize = False

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

env.reset()

#env.save_replay("rulebase_replay")

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

# Action part
_NO_OP = actions.FUNCTIONS.no_op.id

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_HOLDPOSITION_QUICK = actions.FUNCTIONS.HoldPosition_quick.id
_NOT_QUEUED = [0]
_QUEUED = [1]

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_SMART_MINIMAP = actions.FUNCTIONS.Smart_minimap.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_BUILD_COMMANDCENTER_SCREEN = actions.FUNCTIONS.Build_CommandCenter_screen.id
_BUILD_SUPPLYDEPOT_SCREEN = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS_SCREEN = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_TECHLAB_SCREEN = actions.FUNCTIONS.Build_TechLab_screen.id
_BUILD_TECHLAB_QUICK = actions.FUNCTIONS.Build_TechLab_quick.id
_BUILD_REACTOR_QUICK = actions.FUNCTIONS.Build_Reactor_quick.id
_BUILD_REACTOR_SCREEN = actions.FUNCTIONS.Build_Reactor_screen.id
_BUILD_BUNKER_SCREEN = actions.FUNCTIONS.Build_Bunker_screen.id
_BUILD_STARPORT_SCREEN = actions.FUNCTIONS.Build_Starport_screen.id
_BUILD_FACTORY_SCREEN = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_ARMORY_SCREEN = actions.FUNCTIONS.Build_Armory_screen.id
_BUILD_ENGINNERINGBAY_SCREEN = actions.FUNCTIONS.Build_EngineeringBay_screen.id

_TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER_QUICK = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_SCV_QUICK = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_SIEGETANK_QUICK = actions.FUNCTIONS.Train_SiegeTank_quick.id
_TRAIN_MEDIVAC_QUICK = actions.FUNCTIONS.Train_Medivac_quick.id
_TRAIN_REAPER_QUICK = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_HELLION_QUICK = actions.FUNCTIONS.Train_Hellion_quick.id
_TRAIN_VIKINGFIGHTER_QUICK = actions.FUNCTIONS.Train_VikingFighter_quick.id

_RETURN_SCV_QUICK = actions.FUNCTIONS.Harvest_Return_SCV_quick.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_HARVEST_GATHER_SCV_SCREEN = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_LIFT_QUICK = actions.FUNCTIONS.Lift_quick.id
_MORPH_SUPPLYDEPOT_LOWER_QUICK = actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id
_MORPH_SUPPLYDEPOT_RAISE_QUICK = actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick.id
_MORPH_ORBITALCOMMAND_QUICK = actions.FUNCTIONS.Morph_OrbitalCommand_quick.id
_LAND_SCREEN = actions.FUNCTIONS.Land_screen.id
_CANCEL_LAST_QUICK = actions.FUNCTIONS.Cancel_Last_quick.id
_RALLY_WORKERS_SCREEN = actions.FUNCTIONS.Rally_Workers_screen.id
_HARVEST_RETURN_QUICK = actions.FUNCTIONS.Harvest_Return_quick.id
_PATROL_SCREEN = actions.FUNCTIONS.Patrol_screen.id
_EFFECT_COOLDOWNMULE_SCREEN = actions.FUNCTIONS.Effect_CalldownMULE_screen.id
_BUILD_QUEUE = actions.FUNCTIONS.build_queue.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_EFFECT_KD8CHARGE_SCREEN = actions.FUNCTIONS.Effect_KD8Charge_screen.id
_HALT_QUICK = actions.FUNCTIONS.Halt_quick.id

_RESEARCH_STIMPACK_QUICK = actions.FUNCTIONS.Research_Stimpack_quick.id
_RESEARCH_COMBATSHIELD_QUICK = actions.FUNCTIONS.Research_CombatShield_quick.id
_UNLOAD = actions.FUNCTIONS.unload.id

action_type_list = [_BUILD_SUPPLYDEPOT_SCREEN, _BUILD_BARRACKS_SCREEN, _BUILD_REFINERY_SCREEN, _BUILD_TECHLAB_SCREEN, _BUILD_COMMANDCENTER_SCREEN, 
                        _BUILD_REACTOR_QUICK, _BUILD_BUNKER_SCREEN, _BUILD_STARPORT_SCREEN, _BUILD_FACTORY_SCREEN, _HALT_QUICK, _RESEARCH_COMBATSHIELD_QUICK,
                        _TRAIN_MARINE_QUICK, _TRAIN_MARAUDER_QUICK, _TRAIN_SCV_QUICK, _TRAIN_SIEGETANK_QUICK, _TRAIN_MEDIVAC_QUICK, _TRAIN_REAPER_QUICK,
                        _RETURN_SCV_QUICK, _HARVEST_GATHER_SCREEN, _HARVEST_GATHER_SCV_SCREEN, _PATROL_SCREEN, _SELECT_UNIT, _HOLDPOSITION_QUICK,
                        _SELECT_CONTROL_GROUP, _LIFT_QUICK, _MORPH_SUPPLYDEPOT_LOWER_QUICK, _LAND_SCREEN, _BUILD_TECHLAB_QUICK, _RESEARCH_STIMPACK_QUICK,
                        _ATTACK_SCREEN, _ATTACK_MINIMAP, _SMART_SCREEN, _SMART_MINIMAP, _MORPH_ORBITALCOMMAND_QUICK, _BUILD_ENGINNERINGBAY_SCREEN,
                        _SELECT_POINT, _SELECT_RECT, _SELECT_IDLE_WORKER, _SELECT_CONTROL_GROUP, _SELECT_ARMY, _BUILD_ARMORY_SCREEN, _BUILD_REACTOR_SCREEN,
                        _MOVE_SCREEN, _MOVE_CAMERA, _CANCEL_LAST_QUICK, _RALLY_WORKERS_SCREEN, _HARVEST_RETURN_QUICK, _TRAIN_HELLION_QUICK, 
                        _EFFECT_COOLDOWNMULE_SCREEN, _MORPH_SUPPLYDEPOT_RAISE_QUICK, _BUILD_QUEUE, _EFFECT_KD8CHARGE_SCREEN, _UNLOAD,
                        _TRAIN_VIKINGFIGHTER_QUICK]

home_upgrade_array = np.zeros(89)
away_upgrade_array = np.zeros(89)
class Agent(object):
  """Demonstrates agent interface.

  In practice, this needs to be instantiated with the right neural network
  architecture.
  """
  def __init__(self, race='Terran', batch_size=1):
    self.home_race = race
    self.away_race = 'Terran'

    self.build_order = []
    self.supply_depot_built = False

    self.scv_selected = False
    self.scv_return = False

    self.train_marine_flag = False
    self.train_marauder_flag = False

    self.build_supply_depot_flag = False
    self.build_barracks_flag = False
    self.build_refinery_flag = False
    self.build_techlab_flag = False

    self.marine_selected = False
    self.marauder_selected = False
    self.army_selected = False

    self.first_attack = False
    self.second_attack = False

    self.action_phase = 0
    self.previous_action = None
    self.selected_unit = []

    self.agent_model = None
    self.batch_size = batch_size
  
  def make_model(self):
      feature_screen = tf.keras.Input(shape=[27, 128, 128])
      embedded_feature_units = tf.keras.Input(shape=[512,464])
      core_prev_state = (tf.keras.Input(shape=[128]), tf.keras.Input(shape=[128]))
      embedded_scalar = tf.keras.Input(shape=[307])
      scalar_context = tf.keras.Input(shape=[842])

      map_, embedded_spatial = SpatialEncoder(img_height=128, img_width=128, channel=27)(feature_screen)
      embedded_entity, entity_embeddings = EntityEncoder(464, 8)(embedded_feature_units)
      lstm_output, final_memory_state, final_carry_state = Core(128)(core_prev_state, embedded_entity, embedded_spatial, embedded_scalar)
      action_type_logits, action_type, autoregressive_embedding_action = ActionTypeHead(len(action_type_list))(lstm_output, scalar_context)
      '''
      agent_model = tf.keras.Model(
          inputs=[feature_screen, embedded_feature_units, core_prev_state, embedded_scalar, scalar_context],
          outputs=[action_type_logits, action_type, final_memory_state, final_carry_state, lstm_output, autoregressive_embedding_action]
      )
      '''
      selected_units_logits, selected_units, autoregressive_embedding_select = SelectedUnitsHead()(autoregressive_embedding_action, 
                                                                                                                         action_type, 
                                                                                                                         entity_embeddings)
      target_unit_logits, target_unit = TargetUnitHead()(autoregressive_embedding_select, action_type, entity_embeddings)

      target_location_logits, target_location = LocationHead()(autoregressive_embedding_select, action_type, map_)
      agent_model = tf.keras.Model(
          inputs=[feature_screen, embedded_feature_units, core_prev_state, embedded_scalar, scalar_context],
          outputs=[action_type_logits, action_type, selected_units_logits, selected_units, target_unit_logits, target_unit, target_location_logits, target_location, 
                     final_memory_state, final_carry_state, autoregressive_embedding_action]
      )
      
      #agent_model.summary()

      self.agent_model = agent_model
  
  def step(self, observation, core_state):
    global home_upgrade_array
    global away_upgrade_array
    global previous_action

    """Performs inference on the observation, given hidden state last_state."""
    # We are omitting the details of network inference here.
    feature_screen = observation[3]['feature_screen']
    # feature_screen.shape: (27, 128, 128)

    feature_minimap = observation[3]['feature_minimap']
    feature_units = observation[3]['feature_units']
    feature_player = observation[3]['player']
    score_by_category = observation[3]['score_by_category']
    game_loop = observation[3]['game_loop']
    available_actions = observation[3]['available_actions']
    # available_actions: [  0   1   2   3   4 264  12  13 274 549 451 452 453 331 332 333 334  79]

    agent_statistics = get_agent_statistics(score_by_category)
    # agent_statistics.shape: (55,)

    race = get_race_onehot(self.home_race, self.away_race)
    # race.shape: (10,)

    time = get_gameloop_obs(game_loop)
    #time.shape : (64,)

    upgrade_value = get_upgrade_obs(feature_units)
    #print("upgrade_value: " + str(upgrade_value))
    if upgrade_value != -1 and upgrade_value is not None :
      home_upgrade_array[np.where(upgrade_value[0] == 1)] = 1
      away_upgrade_array[np.where(upgrade_value[1] == 1)] = 1

    # home_upgrade_array.shape: (89,)
    # away_upgrade_array.shape: (89,)

    embedded_scalar = np.concatenate((agent_statistics, race, time, home_upgrade_array, away_upgrade_array), axis=0)
    embedded_scalar = np.expand_dims(embedded_scalar, axis=0)
    #print("embedded_scalar.shape: " + str(embedded_scalar.shape))

    cumulative_statistics = observation[3]['score_cumulative'] / 1000.0
    # cumulative_statistics.: [1050    2    0  600  400    0    0    0    0    0    0    0    0]

    cumulative_statistics_array = np.log(cumulative_statistics + 1)

    build_order_array = np.zeros(256)
    if (self.previous_action is not None):
      previous_action = (self.previous_action)

      unit_name = None
      if previous_action == _BUILD_SUPPLYDEPOT_SCREEN:
        #print("_BUILD_SUPPLY_DEPOT true")
        unit_name = 'SupplyDepot'
      elif previous_action == _BUILD_BARRACKS_SCREEN:
        unit_name = 'Barracks'
      elif previous_action == _BUILD_REFINERY_SCREEN:
        unit_name = 'Refinery'
      elif previous_action == _BUILD_TECHLAB_SCREEN:
        unit_name = 'TechLab'
      elif previous_action == _TRAIN_SCV_QUICK:
        unit_name = 'SCV'
      elif previous_action == _TRAIN_MARINE_QUICK:
        unit_name = 'Marine'
      elif previous_action == _TRAIN_MARAUDER_QUICK:
        unit_name = 'Marauder'

      self.previous_action = None
      if unit_name is not None:
        unit_info = int(units_new.get_unit_type(self.home_race, unit_name)[0])
        build_order_array[unit_info] = 1

        if len(self.build_order) <= 20:
          self.build_order.append(build_order_array)

        unit_name = None

    feature_screen = np.expand_dims(feature_screen, axis=0)

    available_actions_array = np.zeros(573)
    available_actions_list = available_actions.tolist()
    for available_action in available_actions_list:
      available_actions_array[available_action] = 1

    scalar_context = np.concatenate((available_actions_array, cumulative_statistics_array, build_order_array), axis=0)
    scalar_context = np.reshape(scalar_context, [1, 842])

    embedded_feature_units = get_entity_obs(feature_units)
    embedded_feature_units = np.reshape(embedded_feature_units, [1,512,464])
    #print("embedded_feature_units: " + str(embedded_feature_units))
    action = [actions.FUNCTIONS.no_op()]

    feature_screen_list = np.vstack([feature_screen])
    embedded_feature_units_list = np.vstack([embedded_feature_units])
    core_state_list = (np.vstack([core_state[0]]), np.vstack([core_state[1]]))  
    embedded_scalar_list = np.vstack([embedded_scalar])
    scalar_context_list = np.vstack([scalar_context])

    #print("scalar_context: " + str(scalar_context))
    #predict_value = self.agent_model([[feature_screen], [embedded_feature_units], [core_state], [embedded_scalar], [scalar_context]])
    predict_value = self.agent_model([feature_screen_list, embedded_feature_units_list, core_state_list, embedded_scalar_list, scalar_context_list])
    #print("predict_value[1]: " + str(predict_value[1]))
    #print("predict_value[3]: " + str(predict_value[3]))

    action_type_logits = predict_value[0].numpy()
    action_type = predict_value[1].numpy()
    selected_units_logits = predict_value[2].numpy()
    selected_units = predict_value[3].numpy()
    target_unit_logits = predict_value[4].numpy()
    target_unit = predict_value[5].numpy()
    target_location_logits = predict_value[6].numpy()
    target_location_x = predict_value[7][0].numpy()
    target_location_y = predict_value[7][1].numpy()
    final_memory_state = predict_value[8].numpy()
    final_carry_state = predict_value[9].numpy()
    
    #print("lstm_output: " + str(lstm_output))
    #print("action_type_logits: " + str(action_type_logits))
    #print("action_type[0]: " + str(action_type[0]))
    #print("selected_units_logits: " + str(selected_units_logits))
    #print("selected_units[0]: " + str(selected_units[0]))
    #print("target_unit_logits: " + str(target_unit_logits))
    #print("target_unit[0]: " + str(target_unit[0]))
    #print("target_location_logits: " + str(target_location_logits))
    #print("target_location_x[0]: " + str(target_location_x[0]))
    #print("target_location_y[0]: " + str(target_location_y[0]))
    #print("final_memory_state.shape: " + str(final_memory_state.shape))
    #print("final_carry_state.shape: " + str(final_carry_state.shape))
    #print("autoregressive_embedding: " + str(autoregressive_embedding))

    #print("")
    
    core_new_state = (final_memory_state, final_carry_state)
    
    #print("action_type_logits.shape[0]: " + str(action_type_logits.shape[0]))
    action = [actions.FUNCTIONS.no_op()]
    selectable_entity_mask = np.zeros(512)
    for idx, feature_unit in enumerate(feature_units):
        selectable_entity_mask[idx] = 1

    if (selected_units < len(feature_units)):
      self.selected_unit.append(feature_units[selected_units[0]])
    else:
      selected_units = None

    #print("target_unit[i]: " + str(target_unit[i]))
    if (target_unit < len(feature_units)):
      target_unit = target_unit[0]
    else:
      target_unit = None

    if self.action_phase == 0 and selected_units is not None and (_SELECT_POINT in available_actions):
      selected_units_info = feature_units[selected_units[0]]
      #print("selected_units_info: " + str(selected_units_info))

      select_point = [selected_units_info.x, selected_units_info.y]
      action = [actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, select_point])]
      self.action_phase = 1
    elif self.action_phase == 1 and action_type_list[action_type[0]] in available_actions:
      position = (target_location_x, target_location_y)
      action = [actions.FunctionCall(action_type_list[action_type[0]], [_NOT_QUEUED, position])]

    self.previous_action = action 
  
    policy_logits = [action_type_logits, selected_units_logits, target_unit_logits, target_location_logits]
    #new_state = core_new_state
    #policy_logits = None
    new_state = core_new_state
    return action, policy_logits, new_state
    
    def unroll(self, trajectory):
      """Unrolls the network over the trajectory.

      The actions taken by the agent and the initial state of the unroll are
      dictated by trajectory.
      """
      # We omit the details of network inference here.
    return policy_logits, baselines

'''
def supervised_update(agent, optimizer, trajectories):
  """Update the agent parameters based on the losses."""

  parameters = agent.get_weights()
  # Compute the forward pass for the window
  policy_logits, _ = agent.unroll(trajectories)

  # Define MLE loss
  mle_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=policy_logits, labels=trajectories[0].target_policy)

  # Define L2 regularization loss
  l2_loss = (tf.reduce_sum([tf.nn.l2_loss(weight) for weight in parameters]))

  loss = mle_loss + 1e-5 * l2_loss
  agent.set_weights(optimizer.minimize(loss))
'''

#replay = Trajectory('/media/kimbring2/Steam/StarCraftII/Replays/', 'Terran', 'Terran', 2500)
#replay.get_random_trajectory()

agent1 = Agent(race='Terran', batch_size=1)
agent1.make_model()

agent2 = Agent()

obs = env.reset()
core_prev_state = (np.zeros([1,128]), np.zeros([1,128]))
for i in range(0, 100):
  print("i: " + str(i))
  # action_1 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]

  action_1, policy_logits_1, new_state_1 = agent1.step(obs[0], core_prev_state)
  #print("new_state_1[0].shape: " + str(new_state_1[0].shape))
  #print("new_state_1[1].shape: " + str(new_state_1[1].shape))
  #print("action_1: " + str(action_1))
  core_prev_state = new_state_1

  #action_2, policy_logits_2, new_state_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]
  obs = env.step([action_1, action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  #print("")

'''
replay_index = 0
core_prev_state = (np.zeros([1, 256]), np.zeros([1, 256]))
optimizer = tf.keras.optimizers.Adam(0.001)
writer = tf.summary.create_file_writer("/media/kimbring2/Steam/AlphaStar_Implementation/tfboard")
for replay_index in range(0, len(replay.home_trajectory) - 1):
  print("replay_index: " + str(replay_index))
  
  #obs = [0, 0, 0, replay.home_trajectory[replay_index][0]]
  acts_human = replay.home_trajectory[replay_index][1]

  online_variables = agent1.agent_model.trainable_variables
  with tf.GradientTape() as tape:
    tape.watch(online_variables)

    action_1, policy_logits_1, new_state_1 = agent1.step(obs[0])
    #action_1, policy_logits_1, new_state_1 = agent1.step(obs, core_prev_state)

    human_action_list = []
    agent_action_logit_list = []
    for act_human in acts_human:
      human_function = str(act_human.function)
      human_arguments = str(act_human.arguments)
      human_action_name = human_function.split('.')[-1]
      human_action_index = action_type_list.index(actions._Functions[human_action_name])
      human_action_list.append(human_action_index)

      agent_action_logit_list.append(policy_logits_1[0])

    y_true = human_action_list
    y_pred = agent_action_logit_list

    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    #print("loss: " + str(scce(y_true, y_pred).numpy()))

    all_losses = scce(y_true, y_pred)
    print("all_losses: " + str(all_losses))
    #tf.summary.scalar('all_losses', all_losses, step=replay_index)

    gradients = tape.gradient(all_losses, online_variables)
    optimizer.apply_gradients(zip(gradients, online_variables))
  #print("train")
  print("")

  #agent1.core_prev_state = new_state_1
  core_prev_state = new_state_1

  action_1 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]
  #print("action_1: " + str(action_1))

  #action_2, policy_logits_2, new_state_2 = agent2.step(obs[1])
  action_2 = [actions.FUNCTIONS.no_op(), actions.FUNCTIONS.no_op()]
  obs = env.step([action_1, action_2])
  #print("env.action_space: " + str(env.action_space))
  #print("obs[0][1]: " + str(obs[0][1]))
  #print("obs[0][0]: " + str(obs[0][0]))
  #print("obs[1][0]: " + str(obs[1][0]))
  #print("")
'''