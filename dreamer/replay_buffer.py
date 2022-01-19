from typing import Dict, Union, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from gym.spaces import Space
from tf_agents.replay_buffers import episodic_replay_buffer

from dreamer.utils import preprocess, quantize

Transition = Dict[str, Union[np.ndarray, dict]]
Batch = Dict[str, np.ndarray]


class ReplayBuffer:
  def __init__(
      self,
      capacity: int,
      observation_space: Space,
      action_space: Space,
      batch_size: int,
      length: int,
      precision: int,
      seed: int
  ):
    dtype = {16: tf.float16, 32: tf.float32}[precision]
    self._sequence_length = length
    data_spec = {
      'observation': tf.TensorSpec(observation_space.shape, tf.uint8),
      'action': tf.TensorSpec(action_space.shape, dtype),
      'reward': tf.TensorSpec((), dtype),
      'terminal': tf.TensorSpec((), dtype)
    }
    self._current_episode = {
      'observation': [],
      'action': [],
      'reward': [],
      'terminal': [],
    }
    self._buffer = episodic_replay_buffer.EpisodicReplayBuffer(
      data_spec,
      seed=seed,
      capacity=capacity,
      buffer_size=1,
      dataset_drop_remainder=True,
      completed_only=False,
      begin_episode_fn=lambda _: True,
      end_episode_fn=lambda _: True
    )
    self.idx = 0
    self._dtype = dtype
    self._dataset = self._buffer.as_dataset(batch_size,
                                            self._sequence_length + 1)
    self._dataset = self._dataset.map(self._preprocess,
                                      tf.data.experimental.AUTOTUNE)
    self._dataset = self._dataset.prefetch(10)

  def _preprocess(self, episode, _):
    episode['observation'] = preprocess(tf.cast(episode['observation'],
                                                self._dtype))
    # Shift observations, terminals and rewards by one timestep, since RSSM
    # always uses the *previous* action and state together with *current*
    # observation to infer the *current* state.
    episode['observation'] = episode['observation'][:, 1:]
    episode['terminal'] = episode['terminal'][:, 1:]
    episode['reward'] = episode['reward'][:, 1:]
    episode['action'] = episode['action'][:, :-1]
    return episode

  def store(self, transition: Transition):
    episode_end = (transition['terminal'] or
                   transition['info'].get('TimeLimit.truncated', False))
    for k, v in self._current_episode.items():
      v.append(transition[k])
    if episode_end:
      self._current_episode['observation'].append(
        transition['next_observation'])
      episode = {k: np.asarray(v) for k, v in self._current_episode.items()}
      episode['observation'] = quantize(episode['observation'])
      new_idx = self._buffer.add_sequence(episode,
                                          tf.constant(self.idx, tf.int64))
      self.idx = int(new_idx)
      self._current_episode = {k: [] for k in self._current_episode.keys()}

  def sample(self, n_batches: int) -> Iterator[Batch]:
    return tfds.as_numpy(self._dataset.take(n_batches))
