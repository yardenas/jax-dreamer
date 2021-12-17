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
            precision=16,
            *input_example: Tuple
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config['clip']),
            optax.scale_by_adam(eps=optimizer_config['eps']),
            optax.scale(-optimizer_config['lr']))
        self.model = model
        self.params = self.model.init(seed, *input_example)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_scaler = {16: jmp.DynamicLossScale(2 ** 15),
                            32: jmp.NoOpLossScale()}[precision]

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


def compute_lambda_values(
        next_values: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray,
        discount: float,
        lambda_: float) -> jnp.ndarray:
    lambda_values = []
    v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
    horizon = next_values.shape[1]
    for t in reversed(range(horizon)):
        td = rewards[:, t] + (1.0 - terminals[:, t]) * (
                1.0 - lambda_) * discount * next_values[:, t]
        v_lambda = td + v_lambda * lambda_ * discount
        lambda_values.append(v_lambda)
    return jnp.asarray(lambda_values).transpose()


def preprocess(image):
    return image / 255.0 - 0.5


def quantize(image):
    return ((image + 0.5) * 255).astype(np.uint8)


def get_mixed_precision_policy(precision):
    policy = ('params=float32,compute=float' + str(precision) +
              ',output=float' + str(precision))
    return jmp.get_policy(policy)


@functools.partial(jax.jit, static_argnums=3)
def evaluate_model(observations, actions, key, model, model_params, precision):
    length = min(len(observations) + 1, 50)
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
    out = (observations[None], infered_decoded.mean(), generated_decoded.mean())
    out = jax.tree_map(lambda x: ((x + 0.5) * 255).astype(jnp.uint8), out)
    return out
