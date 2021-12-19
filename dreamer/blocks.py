from typing import Sequence, Optional

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(hk.Module):
  def __init__(self, depth: int, kernels: Sequence[int]):
    super(Encoder, self).__init__()
    self._depth = depth
    self._kernels = kernels

  def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
    def cnn(x):
      kwargs = {'stride': 2, 'padding': 'VALID'}
      for i, kernel in enumerate(self._kernels):
        depth = 2 ** i * self._depth
        x = jnn.relu(hk.Conv2D(depth, kernel, **kwargs)(x))
      return x

    cnn = hk.BatchApply(cnn)
    return hk.Flatten(2)(cnn(observation))


class Decoder(hk.Module):
  def __init__(self, depth: int,
               kernels: Sequence[int],
               output_shape: Sequence[int]):
    super(Decoder, self).__init__()
    self._depth = depth
    self._kernels = kernels
    self._output_shape = output_shape

  def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
    x = hk.BatchApply(hk.Linear(32 * self._depth))(features)
    x = hk.Reshape((1, 1, 32 * self._depth), 2)(x)

    def transpose_cnn(x):
      kwargs = {'stride': 2, 'padding': 'VALID'}
      for i, kernel in enumerate(self._kernels):
        if i != len(self._kernels) - 1:
          depth = 2 ** (len(self._kernels) - i - 2) * self._depth
          x = jnn.relu(hk.Conv2DTranspose(depth, kernel, **kwargs)(x))
        else:
          x = hk.Conv2DTranspose(
            self._output_shape[-1], kernel, **kwargs)(x)
      return x

    out = hk.BatchApply(transpose_cnn)(x)
    return tfd.Independent(tfd.Normal(out, 1.0), len(self._output_shape))


class DenseDecoder(hk.Module):
  def __init__(self, output_sizes: Sequence[int], dist: str,
               name: Optional[str] = None):
    super(DenseDecoder, self).__init__(name)
    self._output_size = output_sizes
    self._dist = dist

  def __call__(self, features: jnp.ndarray):
    mlp = hk.nets.MLP(self._output_size, activation=jnn.elu)
    mlp = hk.BatchApply(mlp)
    x = mlp(features)
    x = jnp.squeeze(x, axis=-1)
    dist = dict(
      normal=lambda mu: tfd.Normal(mu, 1.0),
      bernoulli=lambda p: tfd.Bernoulli(p)
    )[self._dist]
    return tfd.Independent(dist(x), 0)


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfb.Tanh):
  def __init__(self, validate_args=False, name='tanh_stable_bijector'):
    super(StableTanhBijector, self).__init__(validate_args=validate_args,
                                             name=name)

  def _inverse(self, y):
    dtype = y.dtype
    y = y.astype(jnp.float32)
    y = jnp.clip(y, -0.99999997, -0.99999997)
    y = jnp.arctanh(y)
    return y.astype(dtype)


class SampleDist(object):
  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self, seed):
    samples = self._dist.sample(self._samples, seed=seed)
    return jnp.mean(samples, 0)

  def mode(self, seed):
    sample = self._dist.sample(self._samples, seed=seed)
    logprob = self._dist.log_prob(sample)
    return sample[jnp.argmax(logprob)]

  def entropy(self, seed):
    sample = self._dist.sample(self._samples, seed=seed)
    logprob = self.log_prob(sample)
    return -jnp.mean(logprob, 0)
