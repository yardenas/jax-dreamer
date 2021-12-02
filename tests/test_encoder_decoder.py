import unittest
from typing import NamedTuple

import haiku as hk
import jax.numpy as jnp
import jax.random

import dreamer.blocks as blocks


class Fixture(NamedTuple):
    key = jax.random.PRNGKey(42)
    observations: jnp.ndarray = jax.random.normal(key, (5, 50, 64, 64, 3))
    features: jnp.ndarray = jax.random.uniform(key, (5, 50, 32 * 32))


class TestEncoderDecoder(unittest.TestCase):

    def test_forward(self):
        fixture = Fixture()
        key = fixture.key
        encoder = hk.transform(lambda x: blocks.Encoder(32, (4, 4, 4, 4))(x))
        decoder = hk.transform(lambda x: blocks.Decoder(32, (5, 5, 6, 6),
                                                        (64, 64, 3))(x))
        key, subkey = jax.random.split(key)
        enc_params = encoder.init(subkey, fixture.observations)
        features = encoder.apply(enc_params, None, fixture.observations)
        self.assertEqual(features.shape, fixture.features.shape)
        dec_params = decoder.init(subkey, fixture.features)
        image = decoder.apply(dec_params, None, features).mean()
        self.assertEqual(image.shape, fixture.observations.shape)
