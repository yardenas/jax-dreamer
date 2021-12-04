from typing import Tuple

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import dreamer.blocks as b
from dreamer.rssm import RSSM, State, Action, Observation

tfd = tfp.distributions
tfb = tfp.bijectors


class WorldModel(hk.Module):
    def __init__(self, observation_space, config):
        super(WorldModel, self).__init__()
        self.rssm = RSSM(config)
        self.encoder = b.Encoder(config.encoder['depth'],
                                 config.encoder['kernels'])
        self.decoder = b.Decoder(config.decoder['depth'],
                                 config.decoder['kernels'],
                                 observation_space.shape)
        self.reward = b.DenseDecoder(config.reward['output_sizes'] + (1,),
                                     'normal')
        self.terminal = b.DenseDecoder(config.terminal['output_sizes'] + (1,),
                                       'bernoulli')

    def __call__(
            self,
            prev_state: State,
            prev_action: Action,
            observation: Observation
    ) -> Tuple[Tuple[tfd.MultivariateNormalDiag,
                     tfd.MultivariateNormalDiag],
               State]:
        observation = jnp.squeeze(self.encoder(observation[None, None]))
        return self.rssm(prev_state, prev_action, observation)

    def generate_sequence(
            self,
            initial_state: State, policy: hk.Transformed,
            policy_params: hk.Params
    ) -> Tuple[jnp.ndarray, tfd.Normal, tfd.Bernoulli]:
        features = self.rssm.generate_sequence(initial_state,
                                               policy, policy_params)
        reward = self.reward(features)
        terminal = self.terminal(features)
        return features, reward, terminal

    def observe_sequence(
            self,
            observations: Observation, actions: Action
    ) -> Tuple[
        Tuple[tfd.MultivariateNormalDiag,
              tfd.MultivariateNormalDiag],
        jnp.ndarray, tfd.Normal, tfd.Normal, tfd.Bernoulli
    ]:
        observations = self.encoder(observations)
        (prior, posterior), features = self.rssm.observe_sequence(observations,
                                                                  actions)
        reward = self.reward(features)
        terminal = self.terminal(features)
        decoded = self.decode(features)
        return (prior, posterior), features, decoded, reward, terminal

    def decode(self, featuers: jnp.ndarray) -> tfd.Normal:
        return self.decoder(featuers)


class Actor(hk.Module):
    def __init__(self, output_sizes, min_stddev):
        super().__init__()
        self.output_sizes = output_sizes
        self._min_stddev = min_stddev

    def __call__(self, observation):
        mlp = hk.nets.MLP(self.output_sizes)
        mu, stddev = jnp.split(mlp(observation), 2, -1)
        stddev = jnn.softplus(stddev) + self._min_stddev
        multivariate_normal_diag = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=stddev
        )
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(
            multivariate_normal_diag,
            b.StableTanhBijector()
        )
        return b.SampleDist(squashed)


class DoubleCritic(hk.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes

    def __call__(self, observation, action):
        x = jnp.concatenate([observation, action], -1)
        mlp1 = hk.nets.MLP(self.output_sizes)
        mlp2 = hk.nets.MLP(self.output_sizes)

        def to_dist(q_fn):
            mu = jnp.squeeze(q_fn(x), -1)
            return tfd.Normal(loc=mu, scale=1.0)

        return jax.tree_map(to_dist, [mlp1, mlp2])
