from typing import Callable, Tuple, Union

import haiku as hk
import jax.numpy as jnp
import optax

PRNGKey = jnp.ndarray


class Learner:
    def __init__(
            self,
            model: hk.Transformed,
            seed: PRNGKey,
            optimizer_config: dict,
            *input_example: Tuple
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config['clip']),
            optax.scale_by_adam(eps=optimizer_config['eps']),
            optax.scale(-optimizer_config['lr']))
        self.model = model
        self.params = self.model.init(seed, *input_example)
        self.opt_state = self.optimizer.init(self.params)

    @property
    def apply(self) -> Union[Callable, Tuple[Callable]]:
        return self.model.apply


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
    return jnp.ndarray(lambda_values, 1)
