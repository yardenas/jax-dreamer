import functools
import os
import pickle
from collections import defaultdict
from typing import Mapping, Tuple

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
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
            experience: ReplayBuffer,
            logger: TrainingLogger,
            config,
            precision=utils.get_mixed_precision_policy(16),
            prefil_policy=None
    ):
        super(Dreamer, self).__init__()
        self.c = config
        self.rng_seq = hk.PRNGSequence(config.seed)
        dtype = precision.compute_dtype
        self.model = utils.Learner(
            model, next(self.rng_seq), config.model_opt, config.precision,
            observation_space.sample()[None, None].astype(dtype),
            action_space.sample()[None, None].astype(dtype))
        self.precision = precision
        features_example = jnp.concatenate(self.init_state, -1)[None]
        self.actor = utils.Learner(actor, next(self.rng_seq),
                                   config.actor_opt, config.precision,
                                   features_example.astype(dtype))
        self.critic = utils.Learner(
            critic, next(self.rng_seq), config.critic_opt, config.precision,
            features_example[None].astype(dtype))
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
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            self.state = (self.init_state, jnp.zeros_like(self.state[-1]))

    @property
    def init_state(self):
        state = init_state(1, self.c.rssm['stochastic_size'],
                           self.c.rssm['deterministic_size'],
                           self.precision.compute_dtype)
        return jax.tree_map(lambda x: x.squeeze(0), state)

    def update(self):
        reports = defaultdict(float)
        for batch in tqdm(self.experience.sample(next(self.rng_seq),
                                                 self.c.update_steps),
                          leave=False, total=self.c.update_steps):
            (self.model.learning_state,
             self.actor.learning_state,
             self.critic.learning_state,
             report) = self._update(batch, self.model.learning_state,
                                    self.actor.learning_state,
                                    self.critic.learning_state,
                                    next(self.rng_seq))
            # Average training metrics.
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
            key: PRNGKey,
    ) -> Tuple[LearningState, LearningState, LearningState, dict]:
        key, subkey = jax.random.split(key)
        model_state, model_report, features = self.update_model(
            batch, model_state, subkey)
        key, subkey = jax.random.split(key)
        actor_state, actor_report, (
            generated_features, lambda_values
        ) = self.update_actor(features, actor_state, model_state[0],
                              critic_state[0], subkey)
        critic_params, critic_report = self.update_critic(
            generated_features, critic_state, lambda_values)
        report = {**model_report, **actor_report, **critic_report}
        return model_state, actor_state, critic_state, report

    def update_model(
            self,
            batch: Batch,
            state: LearningState,
            key: PRNGKey
    ) -> Tuple[LearningState, dict, jnp.ndarray]:
        params, opt_state, loss_scaler = state

        def loss(params: hk.Params) -> Tuple[float, dict]:
            _, _, infer, _ = self.model.apply
            outputs_infer = infer(params, key, batch['observation'],
                                  batch['action'])
            (prior,
             posterior), features, decoded, reward, terminal = outputs_infer
            kl = jnp.maximum(tfd.kl_divergence(posterior, prior).mean(),
                             self.c.free_kl)
            log_p_obs = decoded.log_prob(batch['observation']).mean()
            log_p_rews = reward.log_prob(batch['reward']).mean()
            log_p_terms = reward.log_prob(batch['terminal']).mean()
            loss_ = self.c.kl_scale * kl - log_p_obs - log_p_rews - log_p_terms
            return loss_scaler.scale(loss_), {
                'agent/model/kl': kl,
                'agent/model/log_p_observation': -log_p_obs,
                'agent/model/log_p_reward': -log_p_rews,
                'agent/model/log_p_terminal': -log_p_terms,
                'features': features
            }

        grads, report = jax.grad(loss, has_aux=True)(params)
        grads = loss_scaler.unscale(grads)
        grads, loss_scaler = utils.nice_grads(grads, loss_scaler)
        updates, new_opt_state = self.model.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params, new_opt_state = jmp.select_tree(grads,
            (new_params, new_opt_state),
            (params, opt_state))
        report['agent/model/grads'] = optax.global_norm(grads)
        return (new_params, new_opt_state, loss_scaler
                ), report, report.pop('features')

    def update_actor(
            self,
            features: jnp.ndarray,
            state: LearningState,
            model_params: hk.Params,
            critic_params: hk.Params,
            key: PRNGKey
    ) -> Tuple[LearningState, dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        params, opt_state, loss_scaler = state
        _, generate_experience, *_ = self.model.apply
        policy = self.actor
        critic = self.critic.apply
        discount = jnp.cumprod(
            self.c.discount * jnp.ones((self.c.imag_horizon - 1,))
        )
        discount = jnp.concatenate([jnp.ones((1,)), discount])

        def loss(params: hk.Params):
            flattened_features = features.reshape((-1, features.shape[-1]))
            generated_features, reward, terminal = generate_experience(
                model_params, key, flattened_features, policy, params)
            values = critic(critic_params, generated_features).mean()
            lambda_values = utils.compute_lambda_values(values[:, 1:],
                                                        reward.mean(),
                                                        terminal.mean(),
                                                        self.c.discount,
                                                        self.c.lambda_)
            loss_ = loss_scaler.scale((-lambda_values * discount[:-1]).mean())
            return loss_, (generated_features, lambda_values)

        (loss_, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
        grads = loss_scaler.unscale(grads)
        grads, loss_scaler = utils.nice_grads(grads, loss_scaler)
        updates, new_opt_state = self.actor.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params, new_opt_state = jmp.select_tree(grads,
                                                    (new_params, new_opt_state),
                                                    (params, opt_state))
        return (new_params, new_opt_state, loss_scaler), {
            'agent/actor/loss': loss_,
            'agent/actor/grads': optax.global_norm(grads)}, aux

    def update_critic(
            self,
            features: jnp.ndarray,
            state: LearningState,
            lambda_values: jnp.ndarray
    ) -> Tuple[LearningState, dict]:
        params, opt_state, loss_scaler = state
        discount = jnp.cumprod(
            self.c.discount * jnp.ones((self.c.imag_horizon - 1,))
        )
        discount = jnp.concatenate([jnp.ones((1,)), discount])

        def loss(params: hk.Params) -> float:
            values = self.critic.apply(params, features[:, :-1])
            targets = jax.lax.stop_gradient(lambda_values)
            return loss_scaler.scale(
                -values.log_prob(targets * discount[:-1]).mean())

        (loss_, grads) = jax.value_and_grad(loss)(params)
        grads = loss_scaler.unscale(grads)
        grads, loss_scaler = utils.nice_grads(grads, loss_scaler)
        updates, new_opt_state = self.critic.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params, new_opt_state = jmp.select_tree(grads,
                                                    (new_params, new_opt_state),
                                                    (params, opt_state))
        return (new_params, new_opt_state, loss_scaler), {
            'agent/critic/loss': loss_,
            'agent/critic/grads': optax.global_norm(grads)}

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
