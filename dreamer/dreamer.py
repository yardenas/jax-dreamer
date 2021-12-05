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

PRNGKey = jnp.ndarray
Observation = np.ndarray
Batch = Mapping[str, np.ndarray]
tfd = tfp.distributions


class Dreamer:
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            model: hk.Transformed,
            actor: hk.Transformed,
            critic: hk.Transformed,
            experience: ReplayBuffer,
            logger: TrainingLogger,
            config
    ):
        super(Dreamer, self).__init__()
        self.rng_seq = hk.PRNGSequence(config.seed)
        self.model = utils.Learner(model, next(self.rng_seq), config.model_opt,
                                   observation_space.sample()[None, None],
                                   action_space.sample()[None, None])
        self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                                   observation_space.sample()[None, None])
        self.critic = utils.Learner(critic, next(self.rng_seq),
                                    config.critic_opt,
                                    observation_space.sample()[None, None])
        self.experience = experience
        self.logger = logger
        self.c = config
        self.training_step = 0
        self._prefill_policy = action_space.sample

    def __call__(self, observation: Observation, training=True):
        if self.training_step <= self.c.prefill and training:
            return self._prefill_policy()
        if self.time_to_update and training:
            self.update()
        if self.time_to_log and training:
            self.logger.log_metrics(self.training_step)
        action = self.policy(observation, self.actor.params,
                             next(self.rng_seq), training)
        return np.clip(action, -1.0, 1.0)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def policy(self, observation: Observation, params: hk.Params,
               rng_key: PRNGKey, training=True) -> jnp.ndarray:
        policy = self.actor.apply(params, observation)
        action = policy.sample(seed=rng_key) if training else policy.mode(
            seed=rng_key)
        return action

    def observe(self, transition):
        self.training_step += 1
        self.experience.store(transition)

    def update(self):
        (self.model.params, self.model.opt_state,
         self.actor.params, self.actor.opt_state,
         self.critic.params, self.critic.opt_state,
         ) = self._update(self.model.params, self.model.opt_state,
                          self.actor.params, self.actor.opt_state,
                          self.critic.params, self.critic.opt_state,
                          self.experience.data,
                          self.experience.episdoe_lengths,
                          next(self.rng_seq))
        self.logger.log_metrics(self.training_step)

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
    ) -> Tuple[hk.Params, optax.OptState,
               hk.Params, optax.OptState,
               hk.Params, optax.OptState]:
        keys = jax.random.split(key, self.c.update_steps)
        for key in keys:
            batch = self.experience.sample(key, data, self.c.sequence_length,
                                           episode_lengths)
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
            for k, v in report.items():
                self.logger[k].update_state(v)
        return (model_params, model_opt_state, actor_params, actor_opt_state,
                critic_params, critic_opt_state)

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
            kl = tfd.kl_divergence(posterior, prior).mean()
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
        policy = self.actor.model
        critic = self.critic.apply
        discount = jnp.cumprod(
            self.c.discount * jnp.ones((self.c.imag_horizon - 1,))
        )
        discount = jnp.concatenate(jnp.ones(()), discount)

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

        (loss_, grads), aux = jax.value_and_grad(loss, has_aux=True)(params)
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

        def loss(params: hk.Params) -> float:
            values = self.critic.apply(params, features)
            targets = jax.lax.stop_gradient(lambda_values)
            return -values.log_prob(targets).mean()

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

    @property
    def time_to_log(self):
        return self.training_step and self.training_step % \
               self.c.log_every == 0
