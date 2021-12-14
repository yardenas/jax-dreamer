import unittest
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from dreamer.models import BayesianWorldModel
from dreamer.rssm import RSSM

tfd = tfp.distributions


class Config(NamedTuple):
    rssm = {'deterministic_size': 32, 'stochastic_size': 4, 'hidden': 5}
    imag_horizon = 12
    encoder = {'depth': 32, 'kernels': (4, 4, 4, 4)}
    decoder = {'depth': 32, 'kernels': (5, 5, 6, 6)}
    reward = {'output_sizes': (400, 400, 400, 400)}
    terminal = {'output_sizes': (400, 400, 400, 400)}
    rssm_posterior = {'stddev': 1.0, 'learnable': True}
    rssm_prior = {'stddev': 1.0, 'learnable': False}
    rssm_posterior_samples = 10


CONFIG = Config()


class Fixture:
    SEED = jax.random.PRNGKey(42)
    POLICY = hk.without_apply_rng(
        hk.transform(lambda x: tfd.Normal(hk.Linear(1)(x), 1.0)))
    SEED, subkey = jax.random.split(SEED)
    POLICY_PARAMS = POLICY.init(subkey, jnp.zeros((1, 36)))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_OBSERVATIONS = jax.random.uniform(subkey, (3, 15, 64, 64, 3))
    SEED, subkey = jax.random.split(SEED)
    DUMMY_ACTIONS = jax.random.uniform(subkey, (3, 14, 1))
    SEED, subkey1, subkey2 = jax.random.split(SEED, 3)
    DUMMY_STATE = (jax.random.uniform(subkey1, (3, 4,)),
                   jax.random.uniform(subkey2, (3, 32,)))
    DUMMY_FEATURES = jnp.concatenate(DUMMY_STATE, -1)
    DUMMY_EMBDEDDING = jax.random.uniform(subkey, (3, 15, 1024))


f = Fixture()


def rssm():
    _rssm = RSSM(CONFIG)

    def filter_(prev_state, prev_action, observation):
        return _rssm(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy,
                          policy_params, actions=None):
        return _rssm.generate_sequence(initial_state, policy,
                                       policy_params, actions)

    def observe_sequence(observations, actions):
        return _rssm.observe_sequence(observations, actions)

    def init(observation, action):
        return _rssm.observe_sequence(observation, action)

    return init, (filter_, generate_sequence, observe_sequence)


RSSM_MODEL = hk.multi_transform(rssm)
RSSM_PARAMS = RSSM_MODEL.init(jax.random.PRNGKey(41), f.DUMMY_EMBDEDDING,
                              f.DUMMY_ACTIONS)


def model():
    _model = BayesianWorldModel(np.ones((64, 64, 3)), RSSM_MODEL, RSSM_PARAMS,
                                CONFIG)

    def filter_state(prev_state, prev_action, observation):
        return _model(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy,
                          policy_params, rssm_params):
        return _model.generate_sequence(initial_state, policy,
                                        policy_params, rssm_params)

    def observe_sequence(observations, actions):
        return _model.observe_sequence(observations, actions)

    def decode(feature):
        return _model.decode(feature)

    def kl():
        return _model.kl()

    def posterior():
        return _model.rssm_posterior()

    def init(observations, actions):
        infer = _model.observe_sequence(observations, actions)
        kl_ = _model.kl()
        postrr = _model.rssm_posterior()
        return infer, kl_, postrr

    return init, (filter_state, generate_sequence, observe_sequence,
                  decode, kl, posterior)


MODEL = hk.multi_transform(model)
PARAMS = MODEL.init(f.SEED, f.DUMMY_OBSERVATIONS, f.DUMMY_ACTIONS)


class TestWorldModel(unittest.TestCase):

    def test_call(self):
        call, *_ = MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        (prior, posterior), state = call(PARAMS, subkey,
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
        _, generate, *_ = MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        features, reward, terminal = generate(PARAMS, subkey,
                                              f.DUMMY_FEATURES,
                                              f.POLICY,
                                              f.POLICY_PARAMS,
                                              RSSM_PARAMS)
        self.assertEqual(features.shape, (3, CONFIG.imag_horizon, 36))
        self.assertEqual(reward.event_shape, ())
        self.assertEqual(tuple(reward.batch_shape), (3, CONFIG.imag_horizon))
        self.assertEqual(terminal.event_shape, ())
        self.assertEqual(tuple(terminal.batch_shape), (3, CONFIG.imag_horizon))

    def test_infer(self):
        _, _, infer, _ = MODEL.apply
        f.SEED, subkey = jax.random.split(f.SEED)
        outputs_infer = infer(PARAMS, subkey,
                              f.DUMMY_OBSERVATIONS,
                              f.DUMMY_ACTIONS)
        (prior, posterior), features, decoded, reward, terminal = outputs_infer
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, (3, 15))
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(features.shape, (3, 15, 36))
        self.assertEqual(reward.event_shape, ())
        self.assertEqual(tuple(reward.batch_shape), (3, 15))
        self.assertEqual(terminal.event_shape, ())
        self.assertEqual(tuple(terminal.batch_shape), (3, 15))
        self.assertEqual(decoded.event_shape, (64, 64, 3))
        self.assertEqual(tuple(decoded.batch_shape), (3, 15))
