import haiku as hk
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class Prior(hk.Module):
    def __init__(self, config):
        super(Prior, self).__init__()
        self.c = config

    def __call__(self, prev_state, prev_action):
        stoch, det = prev_state
        cat = jnp.concatenate([prev_action, stoch], -1)
        x = jax.nn.elu(hk.Linear(self.c['deterministic_size'])(cat))
        x, det = hk.GRU(self.c['deterministic_size'])(x, det)
        x = jax.nn.elu(hk.Linear(self.c['deterministic_size'])(x))
        mean = hk.Linear(self.c['stochastic_size'])(x)
        stddev = jax.nn.softplus(hk.Linear(self.c['stochastic_size'])(x)) + 0.1
        prior = tfd.MultivariateNormalDiag(mean, stddev)
        sample = prior.sample(seed=hk.next_rng_key())
        return prior, (sample, det)


class Posterior(hk.Module):
    def __init__(self, config):
        super(Posterior, self).__init__()
        self.c = config

    def __call__(self, prev_state, observation):
        stoch, det = prev_state
        cat = jnp.concatenate([det, observation], -1)
        x = jax.nn.elu(hk.Linear(self.c['deterministic_size'])(cat))
        mean = hk.Linear(self.c['stochastic_size'])(x)
        stddev = jax.nn.softplus(hk.Linear(self.c['stochastic_size'])(x)) + 0.1
        posterior = tfd.MultivariateNormalDiag(mean, stddev)
        sample = posterior.sample(seed=hk.next_rng_key())
        return posterior, (sample, det)


class RSSM(hk.Module):
    def __init__(self, config):
        super(RSSM, self).__init__()
        self.c = config
        self.prior = Prior(config.rssm)
        self.posterior = Posterior(config.rssm)

    def init_state(self, batch_size):
        return (jnp.zeros((batch_size, self.c.rssm['stochastic_size'])),
                jnp.zeros((batch_size, self.c.rssm['deterministic_size'])))

    def __call__(self, prev_state, prev_action, observation):
        prior, state = self.prior(prev_state, prev_action)
        posterior, state = self.posterior(state, observation)
        return (prior, posterior), state

    def generate_sequence(self, initial_state, policy, policy_params):
        def vec(state):
            return jnp.concatenate(state, -1)

        sequence = jnp.zeros(
            (initial_state[0].shape[0],
             self.c.imag_horizon,
             self.c.rssm['stochastic_size'] + self.c.rssm['deterministic_size'])
        )
        state = initial_state
        keys = hk.next_rng_keys(self.c.imag_horizon)
        for t, key in enumerate(keys):
            sample_key, apply_key = jax.random.split(key)
            action = policy.apply(policy_params,
                                  apply_key,
                                  jax.lax.stop_gradient(vec(state))
                                  ).sample(seed=sample_key)
            _, state = self.prior(state, action)
            sequence = sequence.at[:, t].set(vec(state))
        return sequence

    def observe_sequence(self, observations, actions):
        priors, posteriors = [], []
        sequence = jnp.zeros(observations.shape[:2] +
                             (self.c.rssm['stochastic_size'] + self.c.rssm[
                                 'deterministic_size'],))
        state = self.init_state(observations.shape[0])
        for t in range(observations.shape[1]):
            (prior, posterior), state = self.__call__(
                state,
                actions[:, t],
                observations[:, t]
            )
            priors.append((prior.mean(), prior.stddev()))
            posteriors.append((posterior.mean(), posterior.stddev()))
            sequence = sequence.at[:, t].set(jnp.concatenate(state, -1))

        def joint_mvn(dists):
            mvn = tfd.MultivariateNormalDiag(*zip(*dists))
            return tfd.BatchReshape(mvn, observations.shape[:2])

        prior = joint_mvn(priors)
        posterior = joint_mvn(posteriors)
        return (prior, posterior), sequence
