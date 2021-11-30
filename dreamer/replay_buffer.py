from typing import Mapping, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Transition = Mapping[str, Union[np.ndarray, dict]]
Batch = Mapping[str, jnp.ndarray]


class ReplayBuffer(object):
    def __init__(self, capacity, max_episode_length, observation_space,
                 action_space, batch_size, seed):
        self.data = {
            'observation': jnp.full(
                (capacity, max_episode_length + 1) + observation_space.shape,
                jnp.nan, jnp.float32),
            'action': jnp.full(
                (capacity, max_episode_length) + action_space.shape,
                jnp.nan, jnp.float32),
            'reward': jnp.full((capacity, max_episode_length) + (1,),
                               jnp.nan, jnp.float32),
            'terminal': jnp.full((capacity, max_episode_length), (1,),
                                 jnp.nan, jnp.bool_)
        }
        self._episdoe_lengths = jnp.full((capacity,), 0, dtype=jnp.int64)
        self.idx = 0
        self.capacity = capacity
        self.rng_seq = hk.PRNGSequence(seed)
        self._batch_size = batch_size

    def store(self, transition: Transition):
        position = self._episdoe_lengths[self.idx]
        for key, val in self.data.items():
            val[self.idx, position] = transition[key]
            self._episdoe_lengths[self.idx] += 1
        if transition['terminal'] or transition['info'].get(
                'TimeLimit.truncated', False):
            self.data['observation'][self.idx, position] = \
                transition['next_observation']
            self.idx = int((self.idx + 1) % self.capacity)

    def sample(self, samples: int, length: int):
        for _ in range(samples):
            yield self._sample(next(self.rng_seq), self.data,
                               length, self._episdoe_lengths)

    @jax.jit
    def _sample(self,
                key: jnp.ndarray, data: Mapping[str, jnp.ndarray],
                length: int, episode_lengths: jnp.ndarray
                ) -> Batch:
        # Algorithm:
        # 1. Sample episodes by weighting their length, filter too short
        # episodes.
        # 2. Collect sequences from episodes.
        # 3. Repeat 1-2 until `batch size` of sequences is gathered.
        def sample_sequence(key: jnp.ndarray,
                            episode_data: Mapping[str, jnp.ndarray],
                            episode_length: jnp.ndarray,
                            sequence_length: int) -> Tuple[jnp.ndarray, ...]:
            start = jax.random.uniform(
                key, dtype=jnp.int64, minval=0,
                maxval=episode_length - sequence_length + 1)
            end = start + sequence_length
            return tuple(jax.tree_map(lambda x: x[start:end],
                                      episode_data).items())

        episode_count = 0
        # A list of (o,a,r,t) tuples.
        episodes = []
        while episode_count < self._batch_size:
            key, ids_key = jax.random.split(key)
            # Sample uniformly across observations within all episodes which are
            # long enough.
            idxs: jnp.ndarray = self._sample_episode_ids(ids_key,
                                                         episode_lengths,
                                                         length)
            sampled_episods = jax.tree_map(lambda x: x[idxs], data)
            episode_count += idxs.sum()
            key, *sequence_keys = jax.random.split(key, idxs.sum())
            episodes.append(jax.vmap(sample_sequence, (0, None, 0, None))(
                sequence_keys,
                sampled_episods,
                episode_lengths[idxs],
                length))
        observation, action, reward, terminal = map(jnp.stack, zip(*episodes))
        return dict(observation=observation, action=action,
                    reward=reward, terminal=terminal)

    def _sample_episode_ids(self, key: jnp.ndarray,
                            episode_lengths: jnp.ndarray, length: int):
        out = jnp.where(episode_lengths >= length, 1, 0).astype(jnp.int64)
        num_episodes = out.sum()
        logits = jnp.log(episode_lengths[:num_episodes])
        sample = jax.random.categorical(key, logits)
        return (out * sample).astype(jnp.int64)

    def __len__(self):
        return self._episdoe_lengths.sum()
