import haiku as hk
import jax.random
import numpy as np

import dreamer.env_wrappers as env_wrappers
import dreamer.models as models
import train_utils as train_utils
from dreamer.blocks import DenseDecoder, Encoder
from dreamer.dreamer import Dreamer
from dreamer.logger import TrainingLogger
from dreamer.replay_buffer import ReplayBuffer
from dreamer.rssm import RSSM


def create_rssm(config, observation_space, action_space):
    def rssm():
        _rssm = RSSM(config)

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

    # Annoyingly creating an encoder that encodes *dummy* images into embeddings
    # to initialize the RSSM.
    _rssm = hk.multi_transform(rssm)
    dummy_encoder = hk.without_apply_rng(hk.transform(
        lambda x: Encoder(config.encoder['depth'],
                          tuple(config.encoder['kernels'])
                          )(x))
    )
    key = jax.random.PRNGKey(config.seed)
    sample = observation_space.sample()[None, None]
    dummy_encoder_ps = dummy_encoder.init(key, sample)
    rssm_embeddings = dummy_encoder.apply(dummy_encoder_ps, sample)
    rssm_params = _rssm.init(key, rssm_embeddings,
                             action_space.sample()[None, None])
    return _rssm, rssm_params


def create_model(config, observation_space, action_space):
    rssm, rssm_params = create_rssm(config, observation_space, action_space)

    def model():
        _model = models.BayesianWorldModel(
            observation_space, rssm, rssm_params, config)

        def filter_state(prev_state, prev_action, observation):
            return _model(prev_state, prev_action, observation)

        def generate_sequence(initial_state, policy,
                              policy_params, rssm_params=None, actions=None):
            return _model.generate_sequence(initial_state, policy,
                                            policy_params, rssm_params, actions)

        def observe_sequence(observations, actions):
            return _model.observe_sequence(observations, actions)

        def decode(feature):
            return _model.decode(feature)

        def kl():
            return _model.kl()

        def posterior():
            return _model.rssm_posterior()

        def init(observations, actions):
            return _model.observe_sequence(observations, actions)

        return init, (filter_state, generate_sequence, observe_sequence,
                      decode, kl, posterior)

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
    experience = ReplayBuffer(config.replay['capacity'], config.time_limit,
                              environment.observation_space,
                              environment.action_space,
                              config.replay['batch'],
                              config.replay['sequence_length'])
    agent = Dreamer(environment.observation_space,
                    environment.action_space,
                    create_model(config,
                                 environment.observation_space,
                                 environment.action_space),
                    create_actor(config, environment.action_space),
                    create_critic(config), experience,
                    logger, config)
    return agent


if __name__ == '__main__':
    config = train_utils.load_config()
    if not config.jit:
        from jax.config import config as jax_config

        jax_config.update('jax_disable_jit', True)
    environment = env_wrappers.make_env(config.task, config.time_limit,
                                        config.action_repeat, config.seed)
    logger = TrainingLogger(config.log_dir)
    agent = make_agent(config, environment, logger)
    train_utils.train(config, agent, environment, logger)
