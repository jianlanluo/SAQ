import copy
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
import collections
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import distrax

from .jax_utils import extend_and_repeat, next_rng, JaxRNG


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]) -> jnp.ndarray:
    if hasattr(x, 'values'):
        return jnp.concatenate(
            [_flatten_dict(v) for k, v in sorted(x.items())], -1)
    else:
        return x

def update_target_network(main_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y,
        main_params, target_params
    )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param('value', lambda x:self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split('-')]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1e-2, 'fan_in', 'uniform'
                ),
                bias_init=jax.nn.initializers.zeros
            )(x)
        return output

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise')

class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init)(x)
        return jnp.squeeze(x, -1)

    @nn.nowrap
    def rng_keys(self):
        return ('params', )


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0
    use_tanh: bool = True

    def setup(self):
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim, arch=self.arch, orthogonal_init=self.orthogonal_init
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        if self.use_tanh:
            action_distribution = distrax.Transformed(
                action_distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return action_distribution.log_prob(actions)

    def __call__(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        if self.use_tanh:
            action_distribution = distrax.Transformed(
                action_distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        if deterministic:
            samples = mean
            if self.use_tanh:
                samples = jnp.tanh(samples)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=self.make_rng('noise'))

        return samples, log_prob

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise')


class SamplerPolicy(object):

    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=('self', 'deterministic'))
    def act(self, params, rng, observations, deterministic):
        return self.policy.apply(
            params, observations, deterministic, repeat=None,
            rngs=JaxRNG(rng)(self.policy.rng_keys())
        )

    def __call__(self, observations, deterministic=False):
        actions, _ = self.act(self.params, next_rng(), observations, deterministic=deterministic)
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)

    
class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size,
                             kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class StateValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(observations,
                                                   training=training)
        return jnp.squeeze(critic, -1)

class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dims: int = 1

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        inputs = {'states': observations}
        critic = MLP((*self.hidden_dims, self.output_dims),
                     activations=self.activations)(inputs, training=training)
        return critic


class StateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    output_dims: int = 1

    @nn.compact
    def __call__(self, states, actions=None, training: bool = False):
        VmapCritic = nn.vmap(StateActionValue,
                            variable_axes={'params': 0},
                            split_rngs={'params': True},
                            in_axes=None,
                            out_axes=0,
                            axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        output_dims=self.output_dims)(states,
                                                    training)
        return qs
