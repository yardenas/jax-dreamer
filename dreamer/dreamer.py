import functools
import os
import pickle
from collections import defaultdict
from typing import Mapping, Tuple

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

import dreamer.utils as utils
from dreamer.logger import TrainingLogger
from dreamer.replay_buffer import ReplayBuffer
from dreamer.rssm import init_state

PRNGKey = jnp.ndarray
State = Tuple[jnp.ndarray, jnp.ndarray]
Action = jnp.ndarray
Observation = np.ndarray
Batch = Mapping[str, np.ndarray]
tfd = tfp.distributions
LearningState = utils.LearningState


class Dreamer:
  def __init__(
      self,
      observation_space: gym.Space,
      action_space: gym.Space,
      model: hk.MultiTransformed,
      actor: hk.Transformed,
      critic: hk.Transformed,
      optimistic_model: hk.MultiTransformed,
      optimistic_model_params: hk.Params,
      model_likelihood_constraint: hk.Transformed,
      experience: ReplayBuffer,
      logger: TrainingLogger,
      config,
      precision=utils.get_mixed_precision_policy(16),
      prefil_policy=None
  ):
    super(Dreamer, self).__init__()
    self.c = config
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.precision = precision
    dtype = precision.compute_dtype
    self.model = utils.Learner(
      model, next(self.rng_seq), config.model_opt, precision,
      observation_space.sample()[None, None].astype(dtype),
      action_space.sample()[None, None].astype(dtype))
    features_example = jnp.concatenate(self.init_state, -1)[None]
    self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                               precision, features_example.astype(dtype))
    self.critic = utils.Learner(critic, next(self.rng_seq), config.critic_opt,
                                precision, features_example[None].astype(dtype))
    self.optimistic_model = utils.Learner(optimistic_model, next(self.rng_seq),
                                          config.optimistic_model_opt,
                                          precision,
                                          params=optimistic_model_params)
    self.constraint = utils.Learner(model_likelihood_constraint,
                                    next(self.rng_seq), config.constraint_opt,
                                    precision, 0.0)
    self.experience = experience
    self.logger = logger
    self.state = (self.init_state, jnp.zeros(action_space.shape,
                                             precision.compute_dtype))
    self.training_step = 0
    self._prefill_policy = prefil_policy or (
      lambda x: action_space.sample())

  def __call__(self, observation: Observation, training=True):
    if self.training_step <= self.c.prefill and training:
      return self._prefill_policy(observation)
    if self.time_to_update and training:
      self.update()
    action, current_state = self.policy(
      self.state[0], self.state[1], observation, self.model.params,
      self.actor.params, next(self.rng_seq), training)
    self.state = (current_state, action)
    return np.clip(action.astype(np.float32), -1.0, 1.0)

  @functools.partial(jax.jit, static_argnums=(0, 7))
  def policy(
      self,
      prev_state: State,
      prev_action: Action,
      observation: Observation,
      model_params: hk.Params,
      actor_params: hk.Params,
      key: PRNGKey,
      training=True
  ) -> Tuple[jnp.ndarray, State]:
    filter_, *_ = self.model.apply
    key, subkey = jax.random.split(key)
    observation = observation.astype(self.precision.compute_dtype)
    # TODO (yarden): should we use here the optimistic model?
    _, current_state = filter_(model_params, key, prev_state, prev_action,
                               observation)
    features = jnp.concatenate(current_state, -1)[None]
    policy = self.actor.apply(actor_params, features)
    action = policy.sample(seed=key) if training else policy.mode(
      seed=key)
    return action.squeeze(0), current_state

  def observe(self, transition):
    self.training_step += self.c.action_repeat
    self.experience.store(transition)
    if transition['terminal'] or transition['info'].get('TimeLimit.truncated',
                                                        False):
      self.state = (self.init_state, jnp.zeros_like(self.state[-1]))

  @property
  def init_state(self):
    state = init_state(1, self.c.rssm['stochastic_size'],
                       self.c.rssm['deterministic_size'],
                       self.precision.compute_dtype)
    return jax.tree_map(lambda x: x.squeeze(0), state)

  def update(self):
    reports = defaultdict(float)
    for batch in tqdm(self.experience.sample(self.c.update_steps),
                      leave=False, total=self.c.update_steps):
      self.learning_states, report = self._update(dict(batch),
                                                  *self.learning_states,
                                                  key=next(self.rng_seq))
      # Average training metrics across update steps.
      for k, v in report.items():
        reports[k] += float(v) / self.c.update_steps
    self.logger.log_metrics(reports, self.training_step)

  @functools.partial(jax.jit, static_argnums=0)
  def _update(
      self,
      batch: Batch,
      model_state: LearningState,
      actor_state: LearningState,
      critic_state: LearningState,
      optimistic_model_state: LearningState,
      constraint_state: LearningState,
      key: PRNGKey,
  ) -> Tuple[Tuple[LearningState, LearningState, LearningState,
                   LearningState, LearningState], dict]:
    _, key_model, key_actor = jax.random.split(key, 3)
    model_state, model_report, features = self.update_model(batch, model_state,
                                                            key_model)
    states, model_actor_report, aux = self.optimistic_update_actor(
      features, actor_state, optimistic_model_state, constraint_state[0],
      critic_state[0], model_state[0], key_actor)
    actor_state, optimistic_model_state = states
    generated_features, lambda_values, model_log_ps, *_ = aux
    critic_state, critic_report = self.update_critic(
      generated_features, critic_state, lambda_values)
    constraint_state, constraint_report = self.update_constraint(
      model_log_ps, constraint_state)
    report = {**model_report, **model_actor_report, **critic_report,
              **constraint_report}
    states = (model_state, actor_state, critic_state,
              optimistic_model_state, constraint_state)
    return states, report

  def update_model(
      self,
      batch: Batch,
      state: LearningState,
      key: PRNGKey
  ) -> Tuple[LearningState, dict, jnp.ndarray]:
    params, _, loss_scaler = state
    _, _, infer, _, params_kl, _ = self.model.apply

    def loss(params: hk.Params) -> Tuple[float, dict]:
      outputs_infer = infer(params, key, batch['observation'],
                            batch['action'])
      (prior,
       posterior), features, decoded, reward, terminal = outputs_infer
      kl = jnp.maximum(tfd.kl_divergence(posterior, prior).mean(),
                       self.c.free_kl)
      params_kl_ = params_kl(params, None)
      log_p_obs = decoded.log_prob(batch['observation'].astype(jnp.float32)
                                   ).mean()
      log_p_rews = reward.log_prob(batch['reward']).mean()
      log_p_terms = terminal.log_prob(batch['terminal']).mean()
      loss_ = (params_kl_ + self.c.kl_scale * kl -
               log_p_obs - log_p_rews - log_p_terms)
      return loss_scaler.scale(loss_), {
        'agent/model/kl': kl,
        'agent/model/params_kl': params_kl_,
        'agent/model/post_entropy': posterior.entropy().mean(),
        'agent/model/prior_entropy': prior.entropy().mean(),
        'agent/model/log_p_observation': -log_p_obs,
        'agent/model/log_p_reward': -log_p_rews,
        'agent/model/log_p_terminal': -log_p_terms,
        'features': features
      }

    grads, report = jax.grad(loss, has_aux=True)(params)
    new_state = self.model.grad_step(grads, state)
    report['agent/model/grads'] = loss_scaler.scale(optax.global_norm(grads))
    return new_state, report, report.pop('features')

  def optimistic_update_actor(
      self,
      features: jnp.ndarray,
      actor_state: LearningState,
      optimistic_model_state: LearningState,
      constraint_params: hk.Params,
      critic_params: hk.Params,
      model_params: hk.Params,
      key: PRNGKey
  ) -> Tuple[Tuple[LearningState, LearningState], dict,
             Tuple[jnp.ndarray, jnp.ndarray,
                   jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    actor_params, _, actor_loss_scaler = actor_state
    optimistic_model_params, _, model_loss_scaler = optimistic_model_state
    _, generate_experience, *_, rssm_posterior = self.model.apply
    policy = self.actor
    critic = self.critic.apply
    constrain = self.constraint.apply

    def loss(actor_params: hk.Params, optimistic_model_params: hk.Params):
      flattened_features = features.reshape((-1, features.shape[-1]))
      # Generate new experience with model. Model params is used for the
      # Bayesian world model while the optimistic model params is used to
      # parameterize an optimistic RSSM.
      generated_features, reward, terminal = generate_experience(
        model_params, key, flattened_features, policy, actor_params)
      next_values = critic(critic_params, generated_features[:, 1:]).mean()
      lambda_values = utils.compute_lambda_values(
        next_values, reward.mean(), terminal.mean(),
        self.c.discount, self.c.lambda_)
      discount = utils.discount(self.c.discount, self.c.imag_horizon - 1)
      objective = (lambda_values * discount).mean()
      actor_loss = actor_loss_scaler.scale(-objective)
      vec_ps = utils.params_to_vec(optimistic_model_params)
      model_log_ps = rssm_posterior(model_params, None).log_prob(vec_ps)
      constraint, lagrangian = constrain(constraint_params, model_log_ps)
      model_loss = -objective + constraint
      model_loss = model_loss_scaler.scale(model_loss.mean())
      return actor_loss + model_loss, (generated_features, lambda_values,
                                       model_log_ps, actor_loss, model_loss,
                                       lagrangian)

    grads, aux = jax.grad(loss, (0, 1), has_aux=True)(actor_params,
                                                      optimistic_model_params)
    actor_grads, optimistic_model_grads = grads
    new_actor_state = self.actor.grad_step(actor_grads, actor_state)
    new_model_state = self.optimistic_model.grad_step(optimistic_model_grads,
                                                      optimistic_model_state)
    new_state = new_actor_state, new_model_state
    entropy = policy.apply(actor_params, features[:, 0]
                           ).entropy(seed=key).mean()
    return new_state, {
      'agent/actor/loss': actor_loss_scaler.unscale(aux[-3]),
      'agent/actor/grads': actor_loss_scaler.scale(
        optax.global_norm(actor_grads)),
      'agent/actor/entropy': entropy,
      'agent/optimistic_model/loss': model_loss_scaler.unscale(aux[-2]),
      'agent/optimistic_model/grads': model_loss_scaler.scale(
        optax.global_norm(optimistic_model_grads)),
      'agent/optimistic_model/log_p': aux[-4],
      'agent/constraint/lagrangian': aux[-1]
    }, aux

  def update_critic(
      self,
      features: jnp.ndarray,
      state: LearningState,
      lambda_values: jnp.ndarray
  ) -> Tuple[LearningState, dict]:
    params, opt_state, loss_scaler = state

    def loss(params: hk.Params) -> float:
      values = self.critic.apply(params, features[:, :-1])
      targets = jax.lax.stop_gradient(lambda_values)
      discount = utils.discount(self.c.discount, self.c.imag_horizon - 1)
      return loss_scaler.scale(-values.log_prob(targets * discount).mean())

    (loss_, grads) = jax.value_and_grad(loss)(params)
    new_state = self.critic.grad_step(grads, state)
    return new_state, {
      'agent/critic/loss': loss_scaler.unscale(loss_),
      'agent/critic/grads': loss_scaler.scale(optax.global_norm(grads))
    }

  def update_constraint(self,
                        model_log_p: jnp.ndarray,
                        state: LearningState
                        ) -> Tuple[LearningState, dict]:
    params, opt_state, loss_scaler = state

    def loss(params: hk.Params) -> float:
      constraint, _ = self.constraint.apply(params, model_log_p)
      return loss_scaler.scale(-constraint.mean())

    # TODO (yarden): maybe just handwrite this first-order update rule instead?
    (loss_, grads) = jax.value_and_grad(loss)(params)
    new_state = self.constraint.grad_step(grads, state)
    return new_state, {
      'agent/constraint/loss': loss_scaler.unscale(loss_),
      'agent/constraint/grads': loss_scaler.scale(optax.global_norm(grads))
    }

  def write(self, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'checkpoint.pickle'), 'wb') as f:
      pickle.dump({'actor': self.actor,
                   'critics': self.critic,
                   'experience': self.experience,
                   'training_steps': self.training_step}, f)

  def load(self, path):
    with open(os.path.join(path, 'checkpoint.pickle'), 'rb') as f:
      data = pickle.load(f)
    for key, obj in zip(data.keys(), [
      self.actor,
      self.critic,
      self.experience,
      self.training_step
    ]):
      obj = data[key]

  @property
  def time_to_update(self):
    return self.training_step > self.c.prefill and \
           self.training_step % self.c.train_every == 0

  @property
  def learning_states(self):
    return (self.model.learning_state, self.actor.learning_state,
            self.critic.learning_state, self.optimistic_model.learning_state,
            self.constraint.learning_state)

  @learning_states.setter
  def learning_states(self, states):
    (self.model.learning_state, self.actor.learning_state,
     self.critic.learning_state, self.optimistic_model.learning_state,
     self.constraint.learning_state) = states
