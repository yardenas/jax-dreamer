import functools
from typing import Callable, Tuple, Union

import haiku as hk
import jax.numpy as jnp
import jax.random
import jmp
import numpy as np
import optax

PRNGKey = jnp.ndarray
LearningState = Tuple[hk.Params, optax.OptState, jmp.LossScale]


class Learner:
  def __init__(
      self,
      model: Union[hk.Transformed, hk.MultiTransformed],
      seed: PRNGKey,
      optimizer_config: dict,
      precision: jmp.Policy,
      *input_example: Tuple
  ):
    self.optimizer = optax.chain(
      optax.clip_by_global_norm(optimizer_config['clip']),
      optax.scale_by_adam(eps=optimizer_config['eps']),
      optax.scale(-optimizer_config['lr']))
    self.model = model
    self.params = self.model.init(seed, *input_example)
    self.opt_state = self.optimizer.init(self.params)
    self.loss_scaler = {
      jnp.float16: jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15),
                                        period=5000),
      jnp.float32: jmp.NoOpLossScale()
    }[precision.compute_dtype]
    self.precision = precision

  @property
  def apply(self) -> Union[Callable, Tuple[Callable]]:
    return self.model.apply

  @property
  def learning_state(self):
    return self.params, self.opt_state, self.loss_scaler

  @learning_state.setter
  def learning_state(self, state):
    self.params = state[0]
    self.opt_state = state[1]
    self.loss_scaler = state[2]

  def grad_step(self, grads, state: LearningState):
    params, opt_state, loss_scaler = state
    unscaled_grads = loss_scaler.unscale(grads)
    unscaled_grads = self.precision.cast_to_param(unscaled_grads)
    updates, new_opt_state = self.optimizer.update(unscaled_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    grads_finite = jmp.all_finite(unscaled_grads)
    loss_scaler = loss_scaler.adjust(grads_finite)
    new_params, new_opt_state = jmp.select_tree(grads_finite,
                                                (new_params, new_opt_state),
                                                (params, opt_state))
    return new_params, new_opt_state, loss_scaler


def compute_lambda_values(
    next_values: jnp.ndarray,
    rewards: jnp.ndarray,
    terminals: jnp.ndarray,
    discount: float,
    lambda_: float) -> jnp.ndarray:
  v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
  horizon = next_values.shape[1]
  lamda_values = jnp.empty_like(next_values)
  for t in reversed(range(horizon)):
    td = rewards[:, t] + (1.0 - terminals[:, t]) * (
        1.0 - lambda_) * discount * next_values[:, t]
    v_lambda = td + v_lambda * lambda_ * discount
    lamda_values = lamda_values.at[:, t].set(v_lambda)
  return lamda_values


def preprocess(image):
  return image / 255.0 - 0.5


def quantize(image):
  return ((image + 0.5) * 255).astype(np.uint8)


def get_mixed_precision_policy(precision):
  policy = ('params=float32,compute=float' + str(precision) +
            ',output=float' + str(precision))
  return jmp.get_policy(policy)


def discount(factor, length):
  d = np.cumprod(factor * np.ones((length - 1,)))
  d = np.concatenate([np.ones((1,)), d])
  return d


def initializer(name: str) -> hk.initializers.Initializer:
  return {
    'glorot': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    'he': hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
  }[name]


@functools.partial(jax.jit, static_argnums=(3, 5))
def evaluate_model(observations, actions, key, model, model_params, precision):
  length = min(len(observations) + 1, 50)
  observations, actions = jax.tree_map(
    lambda x: x.astype(precision.compute_dtype), (observations, actions))
  _, generate_sequence, infer, decode = model.apply
  key, subkey = jax.random.split(key)
  _, features, infered_decoded, *_ = infer(model_params,
                                           subkey,
                                           observations[None, :length],
                                           actions[None, :length])
  conditioning_length = length // 5
  key, subkey = jax.random.split(key)
  generated, *_ = generate_sequence(
    model_params, subkey,
    features[:, conditioning_length], None, None,
    actions=actions[None, conditioning_length:])
  key, subkey = jax.random.split(key)
  generated_decoded = decode(model_params, subkey, generated)
  out = (observations[None, conditioning_length:length],
         infered_decoded.mean()[:, conditioning_length:length],
         generated_decoded.mean())
  out = jax.tree_map(lambda x: ((x + 0.5) * 255).astype(jnp.uint8), out)
  return out
