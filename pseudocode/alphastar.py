from multiagent import League
from rl import Trajectory, loss_function

from pysc2.env import sc2_env, available_actions_printer
from pysc2.env.environment import StepType
from absl import flags

LOOPS_PER_ACTOR = 1000
BATCH_SIZE = 512
TRAJECTORY_LENGTH = 64

FLAGS = flags.FLAGS
FLAGS(sys.argv)


class SC2Environment:
  """See PySC2 environment."""

  def __init__(self, settings):
    map_name = 'Simple128'
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
    visualize = True

    self.env = sc2_env.SC2Env(
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

  def step(self, home_action, away_action):
    observation = self.env.step([home_action, away_action])
    done_check = obs[0][0]
    is_final = False
    rewards = 0

    if done_check == StepType.LAST:
      is_final = True

    return observation, is_final, rewards

  def reset(self):
    self.env.reset()


class Coordinator:
  """Central worker that maintains payoff matrix and assigns new matches."""
  def __init__(self, league):
    self.league = league

  def send_outcome(self, home_player, away_player, outcome):
    self.league.update(home_player, away_player, outcome)
    if home_player.ready_to_checkpoint():
      self.league.add_player(home_player.checkpoint())


class ActorLoop:
  """A single actor loop that generates trajectories.

  We don't use batched inference here, but it was used in practice.
  """
  def __init__(self, player, coordinator):
    self.player = player
    self.teacher = get_supervised_agent(player.get_race())
    self.environment = SC2Environment()
    self.coordinator = coordinator

  def run(self):
    while True:
      opponent = self.player.get_match()
      trajectory = []
      start_time = time()  # in seconds.
      while time() - start_time < 60 * 60:
        home_observation, away_observation, is_final, z = self.environment.reset()
        student_state = self.player.initial_state()
        opponent_state = opponent.initial_state()
        teacher_state = teacher.initial_state()

        while not is_final:
          student_action, student_logits, student_state = self.player.step(home_observation, student_state)
          # We mask out the logits of unused action arguments.
          action_masks = get_mask(student_action)
          opponent_action, _, _ = opponent.step(away_observation, opponent_state)
          teacher_logits = self.teacher(observation, student_action, teacher_state)

          observation, is_final, rewards = self.environment.step(student_action, opponent_action)
          trajectory.append(Trajectory(
              observation=home_observation,
              opponent_observation=away_observation,
              state=student_state,
              is_final=is_final,
              behavior_logits=student_logits,
              teacher_logits=teacher_logits,
              masks=action_masks,
              action=student_action,
              z=z,
              reward=rewards,
          ))

          if len(trajectory) > TRAJECTORY_LENGTH:
            trajectory = stack_namedtuple(trajectory)
            self.learner.send_trajectory(trajectory)
            trajectory = []
        self.coordinator.send_outcome(student, opponent, self.environment.outcome())


class Learner:
  """Learner worker that updates agent parameters based on trajectories."""
  def __init__(self, player):
    self.player = player
    self.trajectories = []
    self.optimizer = AdamOptimizer(learning_rate=3e-5, beta1=0, beta2=0.99,
                                   epsilon=1e-5)

  def get_parameters():
    return self.player.agent.get_weights()

  def send_trajectory(self, trajectory):
    self.trajectories.append(trajectory)

  def update_parameters(self):
    trajectories = self.trajectories[:BATCH_SIZE]
    self.trajectories = self.trajectories[BATCH_SIZE:]
    loss = loss_function(self.player.agent, trajectories)
    self.player.agent.steps += num_steps(trajectories)
    self.player.agent.set_weights(self.optimizer.minimize(loss))

  @background
  def run(self):
    while True:
      if len(self.trajectories) > BATCH_SIZE:
        self.update_parameters()


def main():
  """Trains the AlphaStar league."""
  league = League(
      initial_agents={
          race: get_supervised_agent(race)
          for race in ("Protoss", "Zerg", "Terran")
      })
  coordinator = Coordinator(league)
  learners = []
  actors = []
  for idx in range(12):
    player = league.get_player(idx)
    learner = Learner(player)
    actors.extend([ActorLoop(player, coordinator) for _ in range(16000)])

  for l in learners:
    l.run()
  for a in actors:
    a.run()

  # Wait for training to finish.
  join()

if __name__ == '__main__':
  main()
