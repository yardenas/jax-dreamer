import functools
from typing import Mapping, Union

import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Space

from dreamer.utils import preprocess

Transition = Mapping[str, Union[np.ndarray, dict]]
Batch = Mapping[str, jnp.ndarray]


class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            max_episode_length: int,
            observation_space: Space,
            action_space: Space,
            batch_size: int,
            length: int
    ):
        device = jax.devices()[0] if (
                len(jax.devices()) < 2) else jax.devices("cpu")[0]
        self.data = {
            'observation': jax.device_put(jnp.full(
                (capacity, max_episode_length + 1) + observation_space.shape,
                jnp.nan, jnp.uint8), device),
            'action': jax.device_put(jnp.full(
                (capacity, max_episode_length) + action_space.shape,
                jnp.nan, jnp.float32), device),
            'reward': jax.device_put(
                jnp.full((capacity, max_episode_length),
                         jnp.nan, jnp.float32), device),
            'terminal': jax.device_put(
                jnp.full((capacity, max_episode_length),
                         jnp.nan, jnp.bool_), device)
        }
        self.episdoe_lengths = jnp.full((capacity,), 0, dtype=jnp.uint32)
        self.idx = 0
        self.capacity = capacity
        self._batch_size = batch_size
        self._length = length

    def store(self, transition: Transition):
        position = self.episdoe_lengths[self.idx]
        for key in self.data.keys():
            data = transition[key] if key != 'observation' else (
                ((transition[key] + 0.5) * 255).astype(jnp.uint8)
            )
            self.data[key] = self.data[key].at[self.idx, position].set(data)
        self.episdoe_lengths = self.episdoe_lengths.at[self.idx].add(1)
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            self.data['observation'] = \
                self.data['observation'].at[self.idx, position].set(
                    transition['next_observation'])
            # If finished the episode too shortly, discard it, since it cannot
            # be used for model learning.
            if position < self._length:
                self.episdoe_lengths = self.episdoe_lengths.at[self.idx].set(0)
            else:
                self.idx = int((self.idx + 1) % self.capacity)

    @functools.partial(jax.jit, static_argnums=0)
    def sample(
            self,
            key: jnp.ndarray,
            data: Mapping[str, jnp.ndarray],
            episode_lengths: jnp.ndarray,
            idx: int
    ) -> Batch:
        # Algorithm:
        # 1. Sample episodes uniformly at random.
        # 2. Sample starting point uniformly and collect sequences from
        # episodes.

        def sample_sequence(key: jnp.ndarray,
                            episode_data: Mapping[str, jnp.ndarray],
                            episode_length: jnp.ndarray
                            ) -> Mapping[str, jnp.ndarray]:
            start = jax.random.randint(
                key, (), 0, episode_length - self._length + 1)
            return jax.tree_map(
                lambda x: jax.lax.dynamic_slice(
                    x, (start,) + (0,) * (len(x.shape) - 1),
                       (self._length,) + x.shape[1:]),
                episode_data)

        key, ids_key = jax.random.split(key)
        idxs = jax.random.randint(ids_key, (self._batch_size,), 0, idx)
        sampled_episods = jax.tree_map(lambda x: x[idxs], data)
        sequence_keys = jax.random.split(key, self._batch_size + 1)[1:]
        sampled_sequences = jax.vmap(sample_sequence, (0, 0, 0))(
            sequence_keys,
            sampled_episods,
            episode_lengths[idxs])
        sampled_sequences['observation'] = preprocess(
            sampled_sequences['observation'], 0.5).astype(jnp.float32)
        return jax.device_put(sampled_sequences, jax.devices()[0])

    def __len__(self):
        return self.episdoe_lengths.sum()
