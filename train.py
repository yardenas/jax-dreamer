import haiku as hk
import numpy as np

import dreamer.env_wrappers as env_wrappers
import dreamer.models as models
import train_utils as train_utils
from dreamer.blocks import DenseDecoder
from dreamer.dreamer import Dreamer
from dreamer.logger import TrainingLogger
from dreamer.replay_buffer import ReplayBuffer
from dreamer.utils import get_mixed_precision_policy


def create_model(config, observation_space):
    def model():
        _model = models.WorldModel(observation_space, config)

        def filter_state(prev_state, prev_action, observation):
            return _model(prev_state, prev_action, observation)

        def generate_sequence(initial_state, policy,
                              policy_params, actions=None):
            return _model.generate_sequence(initial_state, policy,
                                            policy_params, actions)

        def observe_sequence(observations, actions):
            return _model.observe_sequence(observations, actions)

        def decode(feature):
            return _model.decode(feature)

        def init(observations, actions):
            return _model.observe_sequence(observations, actions)

        return init, (filter_state, generate_sequence, observe_sequence,
                      decode)

    return hk.multi_transform(model)


def create_actor(config, action_space):
    actor = hk.without_apply_rng(hk.transform(
        lambda obs: models.Actor(tuple(config.actor['output_sizes']) +
                                 (2 * np.prod(action_space.shape),),
                                 config.actor['min_stddev'])(obs))
    )
    return actor


def create_critic(config):
    critic = hk.without_apply_rng(hk.transform(
        lambda obs: DenseDecoder(tuple(config.critic['output_sizes']) + (1,),
                                 'normal')(obs)
    ))
    return critic


def make_agent(config, environment, logger):
    precision_policy = get_mixed_precision_policy(config.precision)
    experience = ReplayBuffer(config.replay['capacity'], config.time_limit,
                              environment.observation_space,
                              environment.action_space,
                              config.replay['batch'],
                              config.replay['sequence_length'],
                              precision_policy.compute_dtype)
    agent = Dreamer(environment.observation_space,
                    environment.action_space,
                    create_model(config, environment.observation_space),
                    create_actor(config, environment.action_space),
                    create_critic(config), experience,
                    logger, config, precision_policy)
    return agent


if __name__ == '__main__':
    config = train_utils.load_config()
    if not config.jit:
        from jax.config import config as jax_config

        jax_config.update('jax_disable_jit', True)
    if config.precision == 16:
        policy = get_mixed_precision_policy(config.precision)
        hk.mixed_precision.set_policy(models.WorldModel, policy)
        hk.mixed_precision.set_policy(models.Actor, policy)
        hk.mixed_precision.set_policy(DenseDecoder, policy)
    environment = env_wrappers.make_env(config.task, config.time_limit,
                                        config.action_repeat, config.seed)
    logger = TrainingLogger(config.log_dir)
    agent = make_agent(config, environment, logger)
    train_utils.train(config, agent, environment, logger)
