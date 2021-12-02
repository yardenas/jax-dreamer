from typing import Mapping, Tuple, Union

import functools

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
        self._episdoe_lengths = jnp.full((capacity,), 0, dtype=jnp.uint32)
        self.idx = 0
        self.capacity = capacity
        self.rng_seq = hk.PRNGSequence(seed)
        self._batch_size = batch_size

    def store(self, transition: Transition):
        position = self._episdoe_lengths[self.idx]
        for key in self.data.keys():
            self.data[key] = self.data[key].at[self.idx, position].set(
                transition[key])
        self._episdoe_lengths = self._episdoe_lengths.at[self.idx].add(1)
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            self.data['observation'] = \
                self.data['observation'].at[self.idx, position].set(
                    transition['next_observation'])
            self.idx = int((self.idx + 1) % self.capacity)

    def sample(self, samples: int, length: int):
        for _ in range(samples):
            yield self._sample(next(self.rng_seq), self.data,
                               length, self._episdoe_lengths)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample(
            self,
            key: jnp.ndarray,
            data: Mapping[str, jnp.ndarray],
            length: int, episode_lengths: jnp.ndarray
    ) -> Batch:
        # Algorithm:
        # 1. Sample episodes by weighting their length, filter too short
        # episodes.
        # 2. Collect sequences from episodes.
        # 3. Repeat 1-2 until `batch size` of sequences is gathered.

        def sample_episode_ids(key: jnp.ndarray,
                               episode_lengths: jnp.ndarray, length: int):
            out = jnp.where(episode_lengths >= length, 1, 0).astype(jnp.uint32)
            num_episodes = out.sum()
            logits = episode_lengths[:num_episodes].astype(jnp.float32)
            sample = jax.random.categorical(key, logits,
                                            shape=(self._batch_size,))
            return sample

        def sample_sequence(key: jnp.ndarray,
                            episode_data: Mapping[str, jnp.ndarray],
                            episode_length: jnp.ndarray,
                            sequence_length: int) -> Tuple[jnp.ndarray, ...]:
            start = jax.random.randint(
                key, (1,), 0, episode_length - sequence_length + 1)

            # https: // github.com / google / jax / issues / 5186 and
            # https://github.com/google/jax/issues/101
            # That's the best work-around I could find for dynamic slices
            # (with a static size, but dynamic starting point)
            funky_arange = lambda start, size: start + jnp.cumsum(
                jnp.ones((size,), jnp.int32))

            return jax.tree_map(
                lambda x: x[funky_arange(start, sequence_length)],
                episode_data)

        key, ids_key = jax.random.split(key)
        # Sample uniformly across observations within all episodes which are
        # long enough.
        idxs: jnp.ndarray = sample_episode_ids(ids_key,
                                               episode_lengths,
                                               length)
        sampled_episods = jax.tree_map(lambda x: x[idxs], data)
        sequence_keys = jax.random.split(key, self._batch_size + 1)[1:]
        sampled_sequences = jax.vmap(sample_sequence, (0, 0, 0, None))(
            sequence_keys,
            sampled_episods,
            episode_lengths[idxs],
            length)
        # observation, action, reward, terminal = map(jnp.stack, zip(*episodes))
        return dict()


def __len__(self):
    return self._episdoe_lengths.sum()
