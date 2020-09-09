"""Library for RL losses."""
import collections

import numpy as np


OBSERVATION_FIELDS = [
    'game_seconds',  # Game timer in seconds.
]

ACTION_FIELDS = [
    'action_type',  # Action taken.
    # Other fields, e.g. arguments, repeat, delay, queued.
]

TRAJECTORY_FIELDS = [
    'observation',  # Player observation.
    'opponent_observation',  # Opponent observation, used for value network.
    'state',  # State of the agent (used for initial LSTM state).
    'z',  # Conditioning information for the policy.
    'is_final',  # If this is the last step.
    # namedtuple of masks for each action component. 0/False if final_step of
    # trajectory, or the argument is not used; else 1/True.
    'masks',
    'action',  # Action taken by the agent.
    'behavior_logits',  # namedtuple of logits of the behavior policy.
    'teacher_logits',  # namedtuple of logits of the supervised policy.
    'reward',  # Reward for the agent after taking the step.
]

Trajectory = collections.namedtuple('Trajectory', TRAJECTORY_FIELDS)


def log_prob(actions, logits):
  """Returns the log probability of taking an action given the logits."""
  # Equivalent to tf.sparse_softmax_cross_entropy_with_logits.


def is_sampled(z):
  """Takes a tensor of zs. Returns a mask indicating which z's are sampled."""


def filter_by(action_fields, target):
  """Returns the subset of `target` corresponding to `action_fields`.

  Autoregressive actions are composed of many logits.  We often want to select a
  subset of these logits.

  Args:
    action_fields: One of 'action_type', 'delay', or 'arguments'.
    target: A list of tensors corresponding to the SC2 action spec.
  Returns:
    A list corresponding to a subset of `target`, with only the tensors relevant
    to `action_fields`.
  """


def compute_over_actions(f, *args):
  """Runs f over all elements in the lists composing *args.

  Autoregressive actions are composed of many logits. We run losses functions
  over all sets of logits.
  """
  return sum(f(*a) for a in zip(*args))


def entropy(policy_logits):
  policy = softmax(policy_logits)
  log_policy = logsoftmax(policy_logits)
  ent = np.sum(-policy * log_policy, axis=-1)  # Aggregate over actions.
  # Normalize by actions available.
  normalized_entropy = ent / np.log(policy_logits.shape[-1])
  return normalized_entropy


def entropy_loss(policy_logits, masks):
  """Computes the entropy loss for a set of logits.

  Args:
    policy_logits: namedtuple of the logits for each policy argument.
      Each shape is [..., N_i].
    masks: The masks. Each shape is policy_logits.shape[:-1].
  Returns:
    Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
  """
  return np.mean(compute_over_actions(entropy, policy_logits, masks))


def kl(student_logits, teacher_logits, mask):
  s_logprobs = logsoftmax(student_logits)
  t_logprobs = logsoftmax(teacher_logits)
  teacher_probs = softmax(teacher_logits)
  return teacher_probs * (t_logprobs - s_logprobs) * mask


def human_policy_kl_loss(trajectories, kl_cost, action_type_kl_cost):
  """Computes the KL loss to the human policy.

  Args:
    trajectories: The trajectories.
    kl_cost: A float; the weighting to apply to the KL cost to the human policy.
    action_type_kl_cost: Additional cost applied to action_types for
      conditioned policies.
  Returns:
    Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
  """
  student_logits = trajectories.behavior_logits
  teacher_logits = trajectories.teacher_logits
  masks = trajectories.masks
  kl_loss = compute_over_actions(kl, student_logits, teacher_logits, masks)

  # We add an additional KL-loss on only the action_type for the first 4 minutes
  # of each game if z is sampled.
  game_seconds = trajectories.observation.game_seconds
  action_type_mask = masks.action_type & (game_seconds > 4 * 60)
  action_type_mask = action_type_mask & is_sampled(trajectories.z)
  action_type_loss = kl(student_logits.action_type, teacher_logits.action_type,
                        action_type_mask)
  return (kl_cost * np.mean(kl_loss) +
          action_type_kl_cost * np.mean(action_type_loss))


def lambda_returns(values_tp1, rewards, discounts, lambdas):
  """Computes lambda returns.

  Refer to the following for a similar function:
  https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
  """


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
  """Computes v-trace return advantages.

  Refer to the following for a similar function:
  https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
  """


def td_lambda_loss(baselines, rewards, trajectories):
  discounts = ~trajectories.is_final[:-1]
  returns = lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)
  returns = stop_gradient(returns)
  return 0.5 * np.mean(np.square(returns - baselines[:-1]))


def policy_gradient_loss(logits, actions, advantages, mask):
  """Helper function for computing policy gradient loss for UPGO and v-trace."""
  action_log_prob = log_prob(actions, logits)
  advantages = stop_gradient(advantages)
  return mask * advantages * action_log_prob


def compute_unclipped_logrho(behavior_logits, target_logits, actions):
  """Helper function for compute_importance_weights."""
  return log_prob(actions, target_logits) - log_prob(actions, behavior_logits)


def compute_importance_weights(behavior_logits, target_logits, actions):
  """Computes clipped importance weights."""
  logrho = compute_over_actions(compute_unclipped_logrho, behavior_logits,
                                target_logits, actions)
  return np.minimum(1., np.exp(logrho))


def vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                   action_fields):
  """Computes v-trace policy gradient loss. Helper for split_vtrace_pg_loss."""
  # Remove last timestep from trajectories and baselines.
  trajectories = Trajectory(*tuple(t[:-1] for t in trajectories))
  rewards = rewards[:-1]
  values = baselines[:-1]

  # Filter for only the relevant actions/logits/masks.
  target_logits = filter_by(action_fields, target_logits)
  behavior_logits = filter_by(action_fields, trajectories.behavior_logits)
  actions = filter_by(action_fields, trajectories.actions)
  masks = filter_by(action_fields, trajectories.masks)

  # Compute and return the v-trace policy gradient loss for the relevant subset
  # of logits.
  clipped_rhos = compute_importance_weights(behavior_logits, target_logits,
                                            actions)
  weighted_advantage = vtrace_advantages(clipped_rhos, rewards,
                                         trajectories.discounts, values,
                                         baselines[-1])
  weighted_advantage = [weighted_advantage] * len(target_logits)
  return compute_over_actions(policy_gradient_loss, target_logits,
                              actions, weighted_advantage, masks)


def split_vtrace_pg_loss(target_logits, baselines, rewards, trajectories):
  """Computes the split v-trace policy gradient loss.

  We compute the policy loss (and therefore update, via autodiff) separately for
  the action_type, delay, and arguments. Each of these component losses are
  weighted equally.
  """
  loss = 0.
  loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                         'action_type')
  loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                         'delay')
  loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                         'arguments')
  return loss


def upgo_returns(values, rewards, discounts, bootstrap):
  """Computes the UPGO return targets.

  Args:
    values: Estimated state values. Shape [T, B].
    rewards: Rewards received moving to the next state. Shape [T, B].
    discounts: If the step is NOT final. Shape [T, B].
    bootstrap: Bootstrap values. Shape [B].
  Returns:
    UPGO return targets. Shape [T, B].
  """
  next_values = np.concatenate(
      values[1:], np.expand_dims(bootstrap, axis=0), axis=0)
  # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
  # 1.0) if r_t + V_tp1 > V_t.
  lambdas = (rewards + discounts * next_values) >= values
  # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
  lambdas = np.concatenate(lambdas[1:], np.ones_like(lambdas[-1:]), axis=0)
  return lambda_returns(next_values, rewards, discounts, lambdas)


def split_upgo_loss(target_logits, baselines, trajectories):
  """Computes split UPGO policy gradient loss.

  See split_vtrace_pg_loss docstring for details on split updates.
  See Methods for details on UPGO.
  """
  # Remove last timestep from trajectories and baselines.
  trajectories = Trajectory(*tuple(t[:-1] for t in trajectories))
  values = baselines[:-1]
  returns = upgo_returns(values, trajectories.rewards, trajectories.discounts,
                         baselines[-1])

  # Compute the UPGO loss for each action subset.
  loss = 0.
  for action_fields in ['action_type', 'delay', 'arguments']:
    split_target_logits = filter_by(action_fields, target_logits)
    behavior_logits = filter_by(action_fields, trajectories.behavior_logits)
    actions = filter_by(action_fields, trajectories.actions)
    masks = filter_by(action_fields, trajectories.masks)

    importance_weights = compute_importance_weights(behavior_logits,
                                                    split_target_logits,
                                                    actions)
    weighted_advantage = (returns - values) * importance_weights
    weighted_advantage = [weighted_advantage] * len(split_target_logits)
    loss += compute_over_actions(policy_gradient_loss, split_target_logits,
                                 actions, weighted_advantage, masks)
  return loss


def compute_pseudoreward(trajectories, reward_name):
  """Computes the relevant pseudoreward from trajectories.

  See Methods and detailed_architecture.txt for details.
  """


def loss_function(agent, trajectories):
  """Computes the loss of trajectories given weights."""
  # All ALL_CAPS variables are constants.
  target_logits, baselines = agent.unroll(trajectories)

  loss_actor_critic = 0.
  # We use a number of actor-critic losses - one for the winloss baseline, which
  # outputs the probability of victory, and one for each pseudo-reward
  # associated with following the human strategy statistic z.
  # See the paper methods and detailed_architecture.txt for more details.
  for baseline, costs_and_rewards in zip(baselines,
                                         BASELINE_COSTS_AND_REWARDS):
    pg_cost, baseline_cost, reward_name = costs_and_rewards
    rewards = compute_pseudoreward(trajectories, reward_name)
    loss_actor_critic += (
        baseline_cost * td_lambda_loss(baseline, rewards, trajectories))
    loss_actor_critic += (
        pg_cost *
        split_vtrace_pg_loss(target_logits, baseline, rewards, trajectories))

  loss_upgo = UPGO_WEIGHT * split_upgo_loss(
      target_logits, baselines.winloss_baseline, trajectories)
  loss_he = human_policy_kl_loss(trajectories, KL_COST, ACTION_TYPE_KL_COST)

  loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
  loss_ent = loss_ent * ENT_WEIGHT

  return loss_actor_critic + loss_upgo + loss_he + loss_ent
