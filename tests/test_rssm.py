import unittest
from collections import namedtuple

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from dreamer.rssm import RSSM

tfd = tfp.distributions

Config = namedtuple('Config', ['rssm', 'imag_horizon'])


class Fixture:
    CONFIG = Config(rssm={'deterministic_size': 32, 'stochastic_size': 4},
                    imag_horizon=5)
    SEED = jax.random.PRNGKey(42)
    POLICY = hk.transform(lambda x: tfd.Normal(hk.Linear(1)(x), 1.0))
    SEED, subkey = jax.random.split(SEED)
    POLICY_PARAMS = POLICY.init(subkey, jnp.zeros((1, 36)))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_OBSERVATIONS = jax.random.uniform(subkey, (3, 15, 256))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_ACTIONS = jax.random.uniform(subkey, (3, 14, 1))
    SEED, subkey1, subkey2 = jax.random.split(SEED, 3)
    DUMMY_STATE = (jax.random.uniform(subkey1, (3, 4,)),
                   jax.random.uniform(subkey2, (3, 32,)))


class TestRssm(unittest.TestCase):

    def test_call(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
            RSSM(f.CONFIG)(prev_state, prev_action, observation))
        f.SEED, subkey = jax.random.split(f.SEED)
        params = call.init(subkey, tuple(map(lambda x: x[None, 0],
                                             f.DUMMY_STATE)),
                           f.DUMMY_ACTIONS[None, 0, 0],
                           f.DUMMY_OBSERVATIONS[None, 0, 0])
        (prior, posterior), state = call.apply(
            params, subkey,
            tuple(map(lambda x: x[0], f.DUMMY_STATE)),
            f.DUMMY_ACTIONS[0, 0],
            f.DUMMY_OBSERVATIONS[0, 0])
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, ())
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(state[0].shape, f.DUMMY_STATE[0].shape[-1:])
        self.assertEqual(state[1].shape, f.DUMMY_STATE[1].shape[-1:])

    def test_generate(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
            RSSM(f.CONFIG)(prev_state, prev_action, observation))
        generate = hk.transform(
            lambda initial_state, policy, policy_params:
            RSSM(f.CONFIG).generate_sequence(
                initial_state, policy, policy_params))
        f.SEED, subkey = jax.random.split(f.SEED)
        params = call.init(subkey, tuple(map(lambda x: x[None, 0],
                                             f.DUMMY_STATE)),
                           f.DUMMY_ACTIONS[None, 0, 0],
                           f.DUMMY_OBSERVATIONS[None, 0, 0])
        output = generate.apply(params, subkey,
                                f.DUMMY_STATE,
                                f.POLICY,
                                f.POLICY_PARAMS)
        self.assertEqual(output.shape, (3, 5, 36))

    def test_infer(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
            RSSM(f.CONFIG)(prev_state, prev_action, observation))
        infer = hk.transform(
            lambda observations, actions:
            RSSM(f.CONFIG).observe_sequence(observations, actions))
        f.SEED, subkey = jax.random.split(f.SEED)
        params_infer = infer.init(subkey, f.DUMMY_OBSERVATIONS, f.DUMMY_ACTIONS)
        outputs_infer = infer.apply(params_infer, subkey,
                                    f.DUMMY_OBSERVATIONS,
                                    f.DUMMY_ACTIONS)
        (prior, posterior), outs = outputs_infer
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, (3, 15))
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(outs.shape, (3, 15, 36))
        params_call = call.init(subkey, tuple(map(lambda x: x[None, 0],
                                                  f.DUMMY_STATE)),
                                f.DUMMY_ACTIONS[None, 0, 0],
                                f.DUMMY_OBSERVATIONS[None, 0, 0])
        *_, outs_call = infer.apply(params_call, subkey,
                                    f.DUMMY_OBSERVATIONS,
                                    f.DUMMY_ACTIONS)
        # Verifying that the results are the same, given the same paramters (
        # but a different initialization function).
        self.assertTrue(jnp.equal(outs, outs_call).all())
