import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import dreamer.env_wrappers as env_wrappers
import dreamer.models as models
import train_utils as train_utils
from dreamer.blocks import DenseDecoder, Encoder, Decoder, MeanField
from dreamer.dreamer import Dreamer
from dreamer.logger import TrainingLogger
from dreamer.replay_buffer import ReplayBuffer
from dreamer.rssm import RSSM
from dreamer.utils import get_mixed_precision_policy


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
                      tuple(config.encoder['kernels']),
                      config.initialization)(x))
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
    _model = models.BayesianWorldModel(observation_space, rssm, rssm_params,
                                       config)

    def filter_state(prev_state, prev_action, observation):
      return _model(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy, policy_params,
                          rssm_params=None, actions=None):
      return _model.generate_sequence(initial_state, policy,
                                      policy_params, rssm_params, actions)

    def observe_sequence(observations, actions):
      return _model.observe_sequence(observations, actions)

    def decode(feature):
      return _model.decode(feature)

    def kl():
      return _model.kl()

    def rssm_posterior():
      return _model.rssm_posterior()

    def init(observations, actions):
      return _model.observe_sequence(observations, actions), _model.kl()

    return init, (filter_state, generate_sequence, observe_sequence,
                  decode, kl, rssm_posterior)

  return hk.multi_transform(model)


def create_actor(config, action_space):
  actor = hk.without_apply_rng(hk.transform(
    lambda obs: models.Actor(tuple(config.actor['output_sizes']) +
                             (2 * np.prod(action_space.shape),),
                             config.actor['min_stddev'],
                             config.initialization)(obs))
  )
  return actor


def create_critic(config):
  critic = hk.without_apply_rng(hk.transform(
    lambda obs: DenseDecoder(tuple(config.critic['output_sizes']) + (1,),
                             'normal', config.initialization)(obs)
  ))
  return critic


def create_optimistic_model(config, observation_space, action_space):
  return create_rssm(config, observation_space, action_space)


def create_constraint(config):
  return hk.without_apply_rng(hk.transform(
    lambda log_p:
    models.LikelihoodConstraint(np.log(config.likelihood_threshold))(log_p))
  )


def make_agent(config, environment, logger):
  experience = ReplayBuffer(config.replay['capacity'],
                            environment.observation_space,
                            environment.action_space,
                            config.replay['batch'],
                            config.replay['sequence_length'],
                            config.precision,
                            config.seed)
  precision_policy = get_mixed_precision_policy(config.precision)
  optimistic_rssm, rssm_params = create_optimistic_model(
    config, environment.observation_space, environment.action_space)
  agent = Dreamer(environment.observation_space,
                  environment.action_space,
                  create_model(config, environment.observation_space,
                               environment.action_space),
                  create_actor(config, environment.action_space),
                  create_critic(config),
                  optimistic_rssm,
                  rssm_params,
                  create_constraint(config),
                  experience, logger, config, precision_policy)
  return agent


if __name__ == '__main__':
  config = train_utils.load_config()
  if not config.jit:
    from jax.config import config as jax_config

    jax_config.update('jax_disable_jit', True)
  if config.precision == 16:
    policy = get_mixed_precision_policy(config.precision)
    hk.mixed_precision.set_policy(models.BayesianWorldModel, policy)
    hk.mixed_precision.set_policy(models.Actor, policy)
    hk.mixed_precision.set_policy(DenseDecoder, policy)
    f32_policy = policy.with_output_dtype(jnp.float32)
    hk.mixed_precision.set_policy(Decoder, f32_policy)
    hk.mixed_precision.set_policy(MeanField, f32_policy)
  environment = env_wrappers.make_env(config.task, config.time_limit,
                                      config.action_repeat, config.seed)
  logger = TrainingLogger(config.log_dir)
  agent = make_agent(config, environment, logger)
  train_utils.train(config, agent, environment, logger)
