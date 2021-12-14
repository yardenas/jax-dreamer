from typing import Tuple, Sequence

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from gym.spaces import Space
from tensorflow_probability.substrates import jax as tfp

import dreamer.blocks as b
from dreamer.rssm import State, Action, Observation

tfd = tfp.distributions


class BayesianWorldModel(hk.Module):
    def __init__(self,
                 observation_space: Space,
                 rssm: hk.MultiTransformed,
                 rssm_params: hk.Params,
                 config):
        super(BayesianWorldModel, self).__init__()
        self.rssm = rssm
        self.rssm_posterior = b.MeanField(rssm_params, **config.rssm_posterior)
        self.rssm_prior = b.MeanField(rssm_params, **config.rssm_prior)
        self.encoder = b.Encoder(config.encoder['depth'],
                                 tuple(config.encoder['kernels']))
        self.decoder = b.Decoder(config.decoder['depth'],
                                 tuple(config.decoder['kernels']),
                                 observation_space.shape)
        self.reward = b.DenseDecoder(tuple(config.reward['output_sizes'])
                                     + (1,), 'normal')
        self.terminal = b.DenseDecoder(tuple(config.terminal['output_sizes'])
                                       + (1,), 'bernoulli')
        self._posterior_samples = config.rssm_posterior_samples

    def __call__(
            self,
            prev_state: State,
            prev_action: Action,
            observation: Observation
    ) -> Tuple[Tuple[tfd.MultivariateNormalDiag,
                     tfd.MultivariateNormalDiag],
               State]:
        observation = jnp.squeeze(self.encoder(observation[None, None]))
        # TODO (yarden): How can we do posterior sampling here instead?
        params = self.rssm_posterior.unflatten(
            self.rssm_posterior().mean()
        )
        filter_, *_ = self.rssm.apply
        return filter_(params, hk.next_rng_key(), prev_state, prev_action,
                       observation)

    def generate_sequence(
            self,
            initial_features: jnp.ndarray, actor: hk.Transformed,
            actor_params: hk.Params, rssm_params: hk.Params, actions=None
    ) -> Tuple[jnp.ndarray, tfd.Normal, tfd.Bernoulli]:
        _, generate, *_ = self.rssm.apply
        features = generate(rssm_params, hk.next_rng_key(),
                            initial_features, actor, actor_params, actions)
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
        *_, infer = self.rssm.apply

        def apply_infer(key):
            sampled_params = self.rssm_posterior.unflatten(
                self.rssm_posterior().sample(seed=key)
            )
            return infer(sampled_params, hk.next_rng_key(),
                         observations, actions)

        keys = hk.next_rng_keys(self._posterior_samples)
        outs = jax.vmap(apply_infer, (0,))(keys)
        # Average across parameter posterior samples.
        (priors, posteriors), features = jax.tree_map(lambda x: x.mean(0), outs)

        def joint_mvn(dists):
            mus, stddevs = dists.transpose((1, 2, 0, 3))
            return tfd.MultivariateNormalDiag(mus, stddevs)

        prior = joint_mvn(priors)
        posterior = joint_mvn(posteriors)
        reward = self.reward(features)
        terminal = self.terminal(features)
        decoded = self.decode(features)
        return (prior, posterior), features, decoded, reward, terminal

    def decode(self, featuers: jnp.ndarray) -> tfd.Normal:
        return self.decoder(featuers)

    def kl(self) -> tfd.Distribution:
        return tfd.kl_divergence(self.rssm_posterior(), self.rssm_prior())

    def rssm_posterior(self) -> tfd.MultivariateNormalDiag:
        return self.rssm_posterior()


class Actor(hk.Module):
    def __init__(self, output_sizes: Sequence[int], min_stddev):
        super().__init__()
        self.output_sizes = output_sizes
        self._min_stddev = min_stddev

    def __call__(self, observation):
        mlp = hk.nets.MLP(self.output_sizes, activation=jnn.elu)
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
