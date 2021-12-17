import functools
from typing import Mapping, Union, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Space

from dreamer.utils import preprocess, quantize

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
            length: int,
            dtype=jnp.float32
    ):
        self.data = {
            'observation': np.full(
                (capacity, max_episode_length + 1) + observation_space.shape,
                np.nan, np.uint8),
            'action': np.full(
                (capacity, max_episode_length) + action_space.shape, np.nan,
                np.float32),
            'reward':
                np.full((capacity, max_episode_length), np.nan, np.float32),
            'terminal':
                np.full((capacity, max_episode_length), np.nan, np.bool_)
        }
        self.episode_lengths = np.full((capacity,), 0, dtype=np.uint32)
        self.idx = 0
        self.capacity = capacity
        self._batch_size = batch_size
        self._length = length
        self.dtype = dtype

    def store(self, transition: Transition):
        position = self.episode_lengths[self.idx]
        for key in self.data.keys():
            data = transition[key] if key != 'observation' else (
                quantize(transition[key]))
            self.data[key][self.idx, position] = data
        self.episode_lengths[self.idx] += 1
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            observation = quantize(transition['next_observation'])
            self.data['observation'][self.idx, position] = observation

            # If finished an episode too shortly, discard it, since it cannot
            # be used for model learning.
            if position < self._length:
                self.episode_lengths[self.idx] = 0
            else:
                self.idx = int((self.idx + 1) % self.capacity)

    def sample(self, seed, n_samples) -> Iterator[Batch]:
        # Algorithm:
        # 1. Sample episodes uniformly at random.
        # 2. Sample starting point uniformly and collect sequences from
        # episodes.
        put_cpu = functools.partial(jax.device_put,
                                    device=jax.devices('cpu')[0])
        idxs = np.random.randint(0, self.idx, self._batch_size * n_samples)
        sampled_sequences = self.sample_sequences(put_cpu(self.data),
                                                  put_cpu(self.episode_lengths),
                                                  put_cpu(idxs),
                                                  n_samples, seed)
        sampled_sequences['observation'] = preprocess(
            sampled_sequences['observation'])

        def standardize(item):
            shape = item.shape
            x = item.reshape((n_samples, self._batch_size) + shape[1:])
            return x.astype(self.dtype)

        sampled_sequences = jax.tree_map(standardize, sampled_sequences)
        for sample_id in range(n_samples):
            yield {k: jax.device_put(v[sample_id], jax.devices()[0])
                   for k, v in sampled_sequences.items()}

    def __len__(self):
        return self.episode_lengths.sum()

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def sample_sequences(self, episodes, episode_lengths, idxs,
                         n_samples, seed):
        sampled_episods = jax.tree_map(lambda x: x[idxs], episodes)
        sequence_keys = jax.random.split(
            seed,
            self._batch_size * n_samples + 1)[1:]
        sampled_sequences = jax.vmap(self.sample_sequence, (0, 0, 0))(
            sequence_keys,
            sampled_episods,
            episode_lengths[idxs])
        return sampled_sequences

    def sample_sequence(self, key: jnp.ndarray,
                        episode_data: Mapping[str, np.ndarray],
                        episode_length: np.ndarray
                        ) -> Mapping[str, jnp.ndarray]:
        start = jax.random.randint(
            key, (), 0, episode_length - self._length + 1)
        return jax.tree_map(lambda x: jax.lax.dynamic_slice(
            x, (start,) + (0,) * (len(x.shape) - 1),
               (self._length,) + x.shape[1:]), episode_data)
