import unittest
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from dreamer.models import WorldModel

tfd = tfp.distributions


class Config(NamedTuple):
    rssm = {'deterministic_size': 32, 'stochastic_size': 4}
    imag_horizon = 15
    encoder = {'depth': 32, 'kernels': (4, 4, 4, 4)}
    decoder = {'depth': 32, 'kernels': (5, 5, 6, 6)}
    reward = {'output_sizes': (400, 400, 400, 400)}
    terminal = {'output_sizes': (400, 400, 400, 400)}


CONFIG = Config()


def model():
    _model = WorldModel(np.ones((64, 64, 3)), CONFIG)

    def filter_state(prev_state, prev_action, observation):
        return _model(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy,
                          policy_params):
        return _model.generate_sequence(initial_state, policy,
                                        policy_params)

    def observe_sequence(observations, actions):
        return _model.observe_sequence(observations, actions)

    def decode(feature):
        return _model.decode(feature)

    def init(observations, actions):
        return _model.observe_sequence(observations, actions)

    return init, (filter_state, generate_sequence, observe_sequence,
                  decode)


class Fixture:
    SEED = jax.random.PRNGKey(42)
    POLICY = hk.transform(lambda x: tfd.Normal(hk.Linear(1)(x), 1.0))
    SEED, subkey = jax.random.split(SEED)
    POLICY_PARAMS = POLICY.init(subkey, jnp.zeros((1, 36)))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_OBSERVATIONS = jax.random.uniform(subkey, (3, 15, 64, 64, 3))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_ACTIONS = jax.random.uniform(subkey, (3, 14, 1))
    SEED, subkey1, subkey2 = jax.random.split(SEED, 3)
    DUMMY_STATE = (jax.random.uniform(subkey1, (3, 4,)),
                   jax.random.uniform(subkey2, (3, 32,)))

    MODEL = hk.multi_transform(model)
    PARAMS = MODEL.init(SEED, DUMMY_OBSERVATIONS, DUMMY_ACTIONS)


f = Fixture()


class TestWorldModel(unittest.TestCase):

    def test_call(self):
        call, *_ = f.MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        (prior, posterior), state = call(f.PARAMS, subkey,
                                         tuple(map(lambda x: x[0],
                                                   f.DUMMY_STATE)),
                                         f.DUMMY_ACTIONS[0, 0],
                                         f.DUMMY_OBSERVATIONS[0, 0])
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, ())
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(state[0].shape, f.DUMMY_STATE[0].shape[-1:])
        self.assertEqual(state[1].shape, f.DUMMY_STATE[1].shape[-1:])

    def test_generate(self):
        _, generate, *_ = f.MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        output = generate(f.PARAMS, subkey,
                          f.DUMMY_STATE,
                          f.POLICY,
                          f.POLICY_PARAMS)
        self.assertEqual(output.shape, (3, 5, 36))

    def test_infer(self):
        _, _, infer, _ = f.MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        outputs_infer = infer.apply(f.PARAMS, subkey,
                                    f.DUMMY_OBSERVATIONS,
                                    f.DUMMY_ACTIONS)
        (prior, posterior), outs = outputs_infer
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, (3, 15))
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(outs.shape, (3, 15, 36))
