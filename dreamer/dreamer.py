import functools
import os
import pickle
from typing import Mapping, Tuple

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

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
            prefil_policy=None
    ):
        super(Dreamer, self).__init__()
        self.c = config
        self.rng_seq = hk.PRNGSequence(config.seed)
        self.model = utils.Learner(model, next(self.rng_seq), config.model_opt,
                                   observation_space.sample()[None, None],
                                   action_space.sample()[None, None])
        features_example = jnp.concatenate(self.init_state, -1)[None]
        self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                                   features_example)
        self.critic = utils.Learner(critic, next(self.rng_seq),
                                    config.critic_opt, features_example[None])
        self.experience = experience
        self.logger = logger
        self.state = (self.init_state, jnp.zeros(action_space.shape))
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
        return np.clip(action, -1.0, 1.0)

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
                           self.c.rssm['deterministic_size'])
        return jax.tree_map(lambda x: x.squeeze(0), state)

    def update(self):
        (self.model.params, self.model.opt_state,
         self.actor.params, self.actor.opt_state,
         self.critic.params, self.critic.opt_state), report = self._update(
            self.model.params, self.model.opt_state,
            self.actor.params, self.actor.opt_state,
            self.critic.params, self.critic.opt_state,
            self.experience.data,
            self.experience.episdoe_lengths,
            next(self.rng_seq)
        )
        self.logger.log_metrics(report, self.training_step)

    @functools.partial(jax.jit, static_argnums=0)
    def _update(
            self,
            model_params: hk.Params,
            model_opt_state: optax.OptState,
            actor_params: hk.Params,
            actor_opt_state: optax.OptState,
            critic_params: hk.Params,
            critic_opt_state: optax.OptState,
            data: Mapping[str, jnp.ndarray],
            episode_lengths: jnp.ndarray,
            key: PRNGKey
    ) -> Tuple[Tuple[hk.Params, optax.OptState,
                     hk.Params, optax.OptState,
                     hk.Params, optax.OptState],
               dict]:
        keys = jax.random.split(key, self.c.update_steps)

        def step(carry, key):
            (model_params, model_opt_state, actor_params, actor_opt_state,
             critic_params, critic_opt_state) = carry
            batch = self.experience.sample(key, data, episode_lengths)
            key, subkey = jax.random.split(key)
            model_params, model_report, features = self.update_model(
                batch, model_params, model_opt_state, subkey)
            key, subkey = jax.random.split(key)
            actor_params, actor_report, (
                generated_features, lambda_values
            ) = self.update_actor(features, actor_params, actor_opt_state,
                                  model_params, critic_params, subkey)
            critic_params, critic_report = self.update_critic(
                generated_features, critic_params, critic_opt_state,
                lambda_values)
            report = {**model_report, **actor_report, **critic_report}
            return (model_params, model_opt_state, actor_params,
                    actor_opt_state, critic_params, critic_opt_state), report

        out, reports = jax.lax.scan(
            step,
            init=(model_params, model_opt_state,
                  actor_params,
                  actor_opt_state, critic_params, critic_opt_state),
            xs=keys)
        reports = jax.tree_map(lambda x: x.mean(), reports)
        return out, reports

    def update_model(
            self,
            batch: Batch,
            params: hk.Params,
            opt_state: optax.OptState,
            key: PRNGKey
    ) -> Tuple[hk.Params, dict, jnp.ndarray]:
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
            return loss_, {'agent/model/kl': kl,
                           'agent/model/log_p_observation': log_p_obs,
                           'agent/model/log_p_reward': log_p_rews,
                           'agent/model/log_p_terminal': log_p_terms,
                           'features': features}

        grads, report = jax.grad(loss, has_aux=True)(params)
        updates, new_opt_state = self.model.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        report['agent/model/grads'] = optax.global_norm(grads)
        return new_params, report, report.pop('features')

    def update_actor(
            self,
            features: jnp.ndarray,
            params: hk.Params,
            opt_state: optax.OptState,
            model_params: hk.Params,
            critic_params: hk.Params,
            key: PRNGKey
    ) -> Tuple[hk.Params, dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        _, generate_experience, *_ = self.model.apply
        policy = self.actor
        critic = self.critic.apply
        discount = jnp.cumprod(
            self.c.discount * jnp.ones((self.c.imag_horizon - 1,))
        )
        discount = jnp.concatenate([jnp.ones((1,)), discount])

        def loss(params: hk.Params) -> Tuple[float,
                                             Tuple[jnp.ndarray, jnp.ndarray]]:
            flattened_features = features.reshape((-1, features.shape[-1]))
            generated_features, reward, terminal = generate_experience(
                model_params, key, flattened_features, policy, params)
            values = critic(critic_params, generated_features).mean()
            lambda_values = utils.compute_lambda_values(values[:, 1:],
                                                        reward.mean(),
                                                        terminal.mean(),
                                                        self.c.discount,
                                                        self.c.lambda_)
            return -(
                    lambda_values * discount[:-1]
            ).mean(), (generated_features, lambda_values)

        (loss_, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, new_opt_state = self.actor.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, {'agent/actor/loss': loss_,
                            'agent/actor/grads': optax.global_norm(grads)
                            }, aux

    def update_critic(
            self,
            features: jnp.ndarray,
            params: hk.Params,
            opt_state: optax.OptState,
            lambda_values: jnp.ndarray
    ) -> Tuple[hk.Params, dict]:
        discount = jnp.cumprod(
            self.c.discount * jnp.ones((self.c.imag_horizon - 1,))
        )
        discount = jnp.concatenate([jnp.ones((1,)), discount])

        def loss(params: hk.Params) -> float:
            values = self.critic.apply(params, features[:, :-1])
            targets = jax.lax.stop_gradient(lambda_values)
            return -values.log_prob(targets * discount[:-1]).mean()

        (loss_, grads) = jax.value_and_grad(loss)(params)
        updates, new_opt_state = self.critic.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, {'agent/critic/loss': loss_,
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
