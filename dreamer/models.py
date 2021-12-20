from typing import Tuple, Sequence

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import dreamer.blocks as b
from dreamer.rssm import RSSM, State, Action, Observation

tfd = tfp.distributions


class WorldModel(hk.Module):
  def __init__(self, observation_space, config):
    super(WorldModel, self).__init__()
    self.rssm = RSSM(config)
    self.encoder = b.Encoder(config.encoder['depth'],
                             tuple(config.encoder['kernels']),
                             config.initialization)
    self.decoder = b.Decoder(config.decoder['depth'],
                             tuple(config.decoder['kernels']),
                             observation_space.shape,
                             config.initialization)
    self.reward = b.DenseDecoder(tuple(config.reward['output_sizes'])
                                 + (1,), 'normal',
                                 config.initialization, 'reward')
    self.terminal = b.DenseDecoder(tuple(config.terminal['output_sizes'])
                                   + (1,), 'bernoulli',
                                   config.initialization, 'terminal')

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
      initial_features: jnp.ndarray, actor: hk.Transformed,
      actor_params: hk.Params, actions=None
  ) -> Tuple[jnp.ndarray, tfd.Normal, tfd.Bernoulli]:
    features = self.rssm.generate_sequence(initial_features, actor,
                                           actor_params, actions)
    reward = self.reward(features)
    terminal = self.terminal(features)
    return features, reward, terminal

  # TODO (yarden): Can we infer that state together with the help of
  # the reward and terminal signals and not only the image observation?
  # Maybe by concatenating them to the embeddings after the decoder.
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
  def __init__(self, output_sizes: Sequence[int], min_stddev: float):
    super().__init__()
    self.output_sizes = output_sizes
    self._min_stddev = min_stddev

  def __call__(self, observation):
    mlp = hk.nets.MLP(self.output_sizes, activation=jnn.elu,
                      w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg',
                                                             'uniform'))
    mu, stddev = jnp.split(mlp(observation), 2, -1)
    init_std = np.log(np.exp(5.0) - 1.0).astype(stddev.dtype)
    stddev = jnn.softplus(stddev + init_std) + self._min_stddev
    multivariate_normal_diag = tfd.Normal(5.0 * jnn.tanh(mu / 5.0), stddev)
    # Squash actions to [-1, 1]
    squashed = tfd.TransformedDistribution(multivariate_normal_diag,
                                           b.StableTanhBijector())
    dist = tfd.Independent(squashed, 1)
    return b.SampleDist(dist)
