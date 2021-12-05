import unittest

import gym
import jax
import numpy as np

from dreamer.replay_buffer import ReplayBuffer


def interact(env, episodes, episode_length, buffer):
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    for _ in range(episodes):
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            terminal = done and not info.get('TimeLimit.truncated', False)
            buffer.store(dict(observation=observation,
                              next_observation=next_observation,
                              action=action.astype(np.float32),
                              reward=np.array(reward, np.float32),
                              terminal=np.array(terminal, np.bool),
                              info=info))
            observation = next_observation


class TestReplayBuffer(unittest.TestCase):

    def test_store(self):
        env = gym.make('Pendulum-v1')
        episode_length = 10
        episodes = 3
        capacity = 5
        buffer = ReplayBuffer(capacity, episode_length, env.observation_space,
                              env.action_space, 2, jax.random.PRNGKey(42))
        interact(env, episodes, episode_length, buffer)
        self.assertEqual(buffer.idx, episodes)
        self.assertEqual(buffer.episdoe_lengths[0], episode_length)
        self.assertEqual(buffer.episdoe_lengths[1], episode_length)
        self.assertEqual(buffer.episdoe_lengths[2], episode_length)
        self.assertEqual(buffer.episdoe_lengths[-2], 0)
        self.assertEqual(buffer.episdoe_lengths[-1], 0)

    def test_sample(self):
        from jax.config import config as jax_config
        jax_config.update('jax_disable_jit', True)
        env = gym.make('Pendulum-v1')
        episode_length = 10
        episodes = 3
        capacity = 5
        buffer = ReplayBuffer(capacity, episode_length, env.observation_space,
                              env.action_space, 2, jax.random.PRNGKey(42))
        interact(env, episodes, episode_length, buffer)
        key = jax.random.PRNGKey(43)
        sample = buffer.sample(key, buffer.data, 4, buffer.episdoe_lengths)
        self.assertEqual(sample['observation'].shape[0], 2)
        self.assertEqual(sample['observation'].shape[1], 4)
