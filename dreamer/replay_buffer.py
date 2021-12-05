import functools
from typing import Mapping, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Space

Transition = Mapping[str, Union[np.ndarray, dict]]
Batch = Mapping[str, jnp.ndarray]


class ReplayBuffer(object):
    def __init__(
            self,
            capacity: int,
            max_episode_length: int,
            observation_space: Space,
            action_space: Space,
            batch_size: int,
            seed: jnp.ndarray
    ):
        self.data = {
            'observation': jnp.full(
                (capacity, max_episode_length + 1) + observation_space.shape,
                jnp.nan, jnp.float32),
            'action': jnp.full(
                (capacity, max_episode_length) + action_space.shape,
                jnp.nan, jnp.float32),
            'reward': jnp.full((capacity, max_episode_length) + (1,),
                               jnp.nan, jnp.float32),
            'terminal': jnp.full((capacity, max_episode_length) + (1,),
                                 jnp.nan, jnp.bool_)
        }
        self.episdoe_lengths = jnp.full((capacity,), 0, dtype=jnp.uint32)
        self.idx = 0
        self.capacity = capacity
        self.rng_seq = hk.PRNGSequence(seed)
        self._batch_size = batch_size

    def store(self, transition: Transition):
        position = self.episdoe_lengths[self.idx]
        for key in self.data.keys():
            self.data[key] = self.data[key].at[self.idx, position].set(
                transition[key])
        self.episdoe_lengths = self.episdoe_lengths.at[self.idx].add(1)
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            self.data['observation'] = \
                self.data['observation'].at[self.idx, position].set(
                    transition['next_observation'])
            self.idx = int((self.idx + 1) % self.capacity)

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def sample(
            self,
            key: jnp.ndarray,
            data: Mapping[str, jnp.ndarray],
            length: int,
            episode_lengths: jnp.ndarray
    ) -> Batch:
        # Algorithm:
        # 1. Filter too short episodes.
        # 2. Sample episodes by weighting their length.
        # 3. Sample and collect sequences from episodes.

        def sample_episode_ids(key: jnp.ndarray,
                               episode_lengths: jnp.ndarray,
                               length: int):
            out = jnp.where(episode_lengths >= length, 1, 0).astype(jnp.uint32)
            num_episodes = out.sum()
            logits = episode_lengths[:num_episodes].astype(jnp.float32)
            sample = jax.random.categorical(
                key, logits,
                shape=(self._batch_size,))
            return sample

        def sample_sequence(key: jnp.ndarray,
                            episode_data: Mapping[str, jnp.ndarray],
                            episode_length: jnp.ndarray,
                            sequence_length: int) -> Mapping[str, jnp.ndarray]:
            start = jax.random.randint(
                key, (), 0, episode_length - sequence_length + 1)
            return jax.tree_map(
                lambda x: jax.lax.dynamic_slice(
                    x, (start, 0), (length, x.shape[-1])),
                episode_data)

        key, ids_key = jax.random.split(key)
        # Sample uniformly across observations within all episodes which are
        # long enough.
        idxs: jnp.ndarray = sample_episode_ids(ids_key,
                                               episode_lengths,
                                               length)
        sampled_episods = jax.tree_map(lambda x: x[idxs], data)
        sequence_keys = jax.random.split(
            key, self._batch_size + 1)[1:]
        sampled_sequences = jax.vmap(sample_sequence, (0, 0, 0, None))(
            sequence_keys,
            sampled_episods,
            episode_lengths[idxs],
            length)
        return sampled_sequences

    def __len__(self):
        return self.episdoe_lengths.sum()
