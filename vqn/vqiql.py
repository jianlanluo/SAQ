"""Implementations of algorithms for continuous control."""

import copy
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
import collections
from functools import partial

import distrax
from gym.utils import seeding
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
import flax
from flax import linen as nn
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState


from tqdm import tqdm

from .model import FullyConnectedNetwork, StateActionEnsemble, StateValue
from .jax_utils import next_rng, JaxRNG, collect_jax_metrics, wrap_function_with_rng


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
DataType = Union[np.ndarray, Dict[str, 'DataType']]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
DatasetDict = Dict[str, DataType]

def get_iql_policy_from_model(env, checkpoint_data):
    sampler_policy = IQLSamplerPolicy(checkpoint_data['iql'].actor)
    return sampler_policy


class IQLSamplerPolicy(object):

    def __init__(self, actor):
        self.actor=actor
        rng = jax.random.PRNGKey(24)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        self.rng = rng

    def __call__(self, observations, deterministic=False):
        actions = self.sample_actions(observations)

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions_jit(self.rng, self.actor.apply_fn,
                                          self.actor.params, observations)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    
def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def squared_euclidean_distance(a, b, b2=None, precision=None):
    if b2 is None:
        b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
    a2 = jnp.sum(a**2, axis=1, keepdims=True)
    ab = jnp.matmul(a, b.T, precision=precision)
    d = a2 - 2 * ab + b2
    return d

def entropy_loss_fn(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = jnp.argmax(flat_affinity, axis=-1)
        onehots = jax.nn.one_hot(
            codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype
        )
        onehots = probs - jax.lax.stop_gradient(probs - onehots)
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = jnp.mean(target_probs, axis=0)
    avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
    sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""
    codebook_size: int
    commitment_cost: float
    quantization_cost: float
    entropy_loss_ratio: float = 0.0
    entropy_loss_type: str = "softmax"
    entropy_temperature: float = 1.0

    @nn.compact
    def __call__(self, x, train=False):
        codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            (self.codebook_size, x.shape[-1]),
        )
        codebook = jnp.asarray(codebook, dtype=jnp.float32)
        distances = jnp.reshape(
            squared_euclidean_distance(x, codebook),
            x.shape[:-1] + (self.codebook_size,),
        )
        encoding_indices = jnp.argmin(distances, axis=-1)
        encodings = jax.nn.one_hot(encoding_indices, self.codebook_size, dtype=jnp.float32)
        quantized = self.quantize(encodings)
        result_dict = dict(
            quantized=quantized,
            encoding_indices=encoding_indices,
        )
        if train:
            e_latent_loss = jnp.mean(
                (jax.lax.stop_gradient(quantized) - x) ** 2
            ) * self.commitment_cost
            q_latent_loss = jnp.mean(
                (quantized - jax.lax.stop_gradient(x)) ** 2
            ) * self.quantization_cost
            entropy_loss = 0.0
            if self.entropy_loss_ratio != 0:
                entropy_loss = (
                    entropy_loss_fn(
                        -distances,
                        loss_type=self.entropy_loss_type,
                        temperature=self.entropy_temperature,
                    )
                    * self.entropy_loss_ratio
                )
            e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
            q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
            entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
            loss = e_latent_loss + q_latent_loss + entropy_loss
            result_dict.update(dict(
                loss=loss,
                e_latent_loss=e_latent_loss,
                q_latent_loss=q_latent_loss,
                entropy_loss=entropy_loss,
            ))
            quantized = x + jax.lax.stop_gradient(quantized - x)

        return quantized, result_dict

    def quantize(self, z):
        codebook = jnp.asarray(self.variables["params"]["codebook"], dtype=jnp.float32)
        return jnp.dot(z, codebook)

    def get_codebook(self):
        return jnp.asarray(self.variables["params"]["codebook"], dtype=jnp.float32)

    def decode_ids(self, ids):
        codebook = self.variables["params"]["codebook"]
        return jnp.take(codebook, ids, axis=0)


class ActionVQVAE(nn.Module):
    observation_dim: int
    action_dim: int
    embedding_dim: int
    codebook_size: int
    commitment_cost: float = 1.0
    quantization_cost: float = 1.0
    entropy_loss_ratio: float = 0.0
    entropy_loss_type: str = "softmax"
    entropy_temperature: float = 1.0
    arch: str = '256-256'
    action_only_quantization: bool = False
    reconstruction_loss_type: str = 'l2'

    def setup(self):
        self.encoder = FullyConnectedNetwork(
            output_dim=self.embedding_dim,
            arch=self.arch,
        )
        self.decoder = FullyConnectedNetwork(
            output_dim=self.action_dim,
            arch=self.arch,
        )
        self.vq = VectorQuantizer(
            codebook_size=self.codebook_size,
            commitment_cost=self.commitment_cost,
            quantization_cost=self.quantization_cost,
            entropy_loss_ratio=self.entropy_loss_ratio,
            entropy_loss_type=self.entropy_loss_type,
            entropy_temperature=self.entropy_temperature,
        )
        self.action_prior = FullyConnectedNetwork(
            output_dim=self.codebook_size,
            arch=self.arch,
        )

    @nn.compact
    def __call__(self, observations, actions, train=False):
        if self.action_only_quantization:
            observations = jnp.zeros_like(observations)
        encoder_input = jnp.concatenate([observations, actions], axis=-1)
        encoded_embeddings = self.encoder(encoder_input)
        quantized, vq_result_dict = self.vq(encoded_embeddings, train=train)
        decoder_input = jnp.concatenate([observations, quantized], axis=-1)
        reconstructed = self.decoder(decoder_input)

        if self.reconstruction_loss_type == 'l1':
            reconstruction_loss = jnp.sum(jnp.abs(reconstructed - actions), axis=-1).mean()
        elif self.reconstruction_loss_type == 'l2':
            reconstruction_loss = jnp.sum(jnp.square(reconstructed - actions), axis=-1).mean()
        elif self.reconstruction_loss_type == 'huber':
            reconstruction_loss = jnp.sum(optax.huber_loss(reconstructed, actions), axis=-1).mean()
        else:
            raise ValueError('Unsupported reconstruction loss type!')


        action_prior_logits = self.action_prior(observations)
        action_prior_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            action_prior_logits, vq_result_dict['encoding_indices']
        ))
        action_prior_accuracy = jnp.mean(
            jnp.argmax(action_prior_logits, axis=-1) == vq_result_dict['encoding_indices']
        )
        loss = vq_result_dict['loss'] + reconstruction_loss + action_prior_loss
        result_dict = dict(
            encoded_embeddings=encoded_embeddings,
            reconstructed=reconstructed,
            reconstruction_loss=reconstruction_loss,
            action_prior_loss=action_prior_loss,
            action_prior_accuracy=action_prior_accuracy,
            **vq_result_dict,
        )
        result_dict['loss'] = loss
        return reconstructed, result_dict

    def encode(self, observations, actions):
        if self.action_only_quantization:
            observations = jnp.zeros_like(observations)
        encoder_input = jnp.concatenate([observations, actions], axis=-1)
        encoded_embeddings = self.encoder(encoder_input)
        quantized, vq_result_dict = self.vq(encoded_embeddings, train=False)
        return vq_result_dict['encoding_indices']

    def decode(self, observations, encoding_indices):
        if self.action_only_quantization:
            observations = jnp.zeros_like(observations)
        quantized = self.vq.decode_ids(encoding_indices)
        decoder_input = jnp.concatenate([observations, quantized], axis=-1)
        reconstructed = self.decoder(decoder_input)
        return reconstructed

    def action_prior_logits(self, observations):
        return self.action_prior(observations)

    @nn.nowrap
    def rng_keys(self):
        return ('params', )


class VQSamplerPolicy(object):

    def __init__(self, qf, vqvae, bc_policy, qf_params, vqvae_params, bc_policy_params, sample=True, temperature=1.0, kl_weight=1.0):
        self.qf = qf
        self.vqvae = vqvae
        self.bc_policy = bc_policy
        self.qf_params = qf_params
        self.vqvae_params = vqvae_params
        self.bc_policy_params = bc_policy_params
        self.sample = sample
        self.temperature = temperature
        self.kl_weight = kl_weight

    def update_params(self, qf_params, vqvae_params, bc_policy_params):
        self.qf_params = qf_params
        self.vqvae_params = vqvae_params
        self.bc_policy_params = bc_policy_params
        return self
    
    def _log_softmax(self, logits, axis=-1):
        return jnp.clip(jax.nn.log_softmax(logits, axis=axis), a_min=-8, a_max=None)

    @partial(jax.jit, static_argnames=('self',))
    def act(self, rng, qf_params, vqvae_params, bc_params, observations):
        q_vals = self.qf.apply_fn({'params': qf_params}, observations)

        # compute log pi beta 
        bc_prob = self.bc_policy.apply(bc_params, observations)

        # apply softmax to q policy logits for computing expecation
        q_vals = (q_vals / self.temperature + self._log_softmax(bc_prob)) 

        if self.sample:
            all_actions = vqvae_params['params']['vq']['codebook']

            # sample action from softmax q values
            actions = jnp.expand_dims(random.choice(rng, a=all_actions.shape[0], p=jnp.squeeze(jax.nn.softmax(q_vals, axis=-1))), 0)
        else:
            actions = jnp.argmax(q_vals, axis=-1)

        return self.vqvae.apply(
            vqvae_params, observations, actions,
            method=self.vqvae.decode
        )

    def __call__(self, observations, deterministic):
        del deterministic
        actions = self.act(
            next_rng(), self.qf_params, self.vqvae_params, self.bc_policy_params, observations
        )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


@jax.jit
def select_by_action(q_vals, actions):
    return jnp.squeeze(
        jnp.take_along_axis(
            q_vals, jnp.expand_dims(actions, -1), axis=-1
        ),
        axis=-1
    )

def soft_target_update(critic_params: Params, target_critic_params: Params,
                       tau: float) -> Params:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params,
        target_critic_params)

    return new_target_params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(target_critic: TrainState, value: TrainState, batch: FrozenDict,
             actions: jnp.ndarray, expectile: float,
             critic_reduction: str) -> Tuple[TrainState, Dict[str, float]]:

    q_vals = jnp.squeeze(target_critic.apply_fn({'params': target_critic.params},
                                batch['observations']))
    
    q = select_by_action(q_vals, actions)

    def value_loss_fn(
            value_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        v = value.apply_fn({'params': value_params}, batch['observations'])
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {'value_loss': value_loss, 'v': v.mean()}

    grads, info = jax.grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=grads)

    return new_value, info


def update_q(critic: TrainState, value: TrainState, batch: FrozenDict,
             actions: jnp.ndarray, discount: float, log_prob: jnp.ndarray, 
             kl_divergence_weight: float = 1.0) -> Tuple[TrainState, Dict[str, float]]:
    next_v = value.apply_fn({'params': value.params},
                            batch['next_observations'])

    target_q = batch['rewards'] + discount * batch['masks'] * next_v

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        q_vals = jnp.squeeze(critic.apply_fn({'params': critic_params}, batch['observations'],
                              actions))
        q = select_by_action(q_vals, actions)
        
        critic_loss = ((q - target_q)**2).mean()

        return critic_loss, {'critic_loss': critic_loss, 'q': q.mean(),}

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info



@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_log_prob_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, batch: DatasetDict) -> float:
    dist = actor_apply_fn({'params': actor_params}, batch['observations'])
    log_probs = dist.log_prob(batch['actions'])
    return log_probs.mean()


@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_actions_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                     actor_params: Params,
                     observations: np.ndarray) -> jnp.ndarray:
    dist = actor_apply_fn({'params': actor_params}, observations)
    return dist.mode()


@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)



class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self._actor.apply_fn, self._actor.params,
                                   observations)

        return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params,
                                 batch)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(self._rng, self._actor.apply_fn,
                                          self._actor.params, observations)

        self._rng = rng

        return np.asarray(actions)
    

class VQIQLLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 vqvae_lr: float = 3e-4,
                 embedding_dim: int = 128,
                 codebook_size: int = 64,
                 commitment_cost: float = 1.0,
                 quantization_cost: float = 1.0,
                 entropy_loss_ratio: float = 0.0,
                 entropy_loss_type: str = "softmax",
                 entropy_temperature: float = 1.0,
                 vqvae_arch: str = '512-512',
                 action_only_quantization: bool = False,
                 reconstruction_loss_type: str = 'l2',
                 sample_action: bool = True,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 policy_weight_decay: float = 0.0,
                 qf_weight_decay: float = 0.0,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.9,
                 A_scaling: float = 10.0,
                 critic_reduction: str = 'min',
                 apply_tanh: bool = False,
                 dropout_rate: Optional[float] = None,
                 policy_arch='256-256',
                 policy_log_std_multiplier=1.0,
                 policy_log_std_offset=-1.0,
                 behavior_policy_lr=3e-4,
                 behavior_policy_weight_decay=0.0,
                 kl_divergence_weight=1.0,):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.A_scaling = A_scaling
        self.kl_divergence_weight = kl_divergence_weight

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
    
        if qf_weight_decay != 0:
            critic_optimiser = optax.adamw(learning_rate=critic_lr, weight_decay=qf_weight_decay)
        else:
            critic_optimiser = optax.adam(learning_rate=critic_lr)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=1, output_dims=codebook_size)
        critic_params = critic_def.init(critic_key, observations,
                                        actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=critic_optimiser)
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        value_params = value_def.init(value_key, observations)['params']
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=optax.adam(learning_rate=value_lr))

        self._rng = rng
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._value = value

        self.observation_dim = observations.shape[-1]
        self.action_dim = actions.shape[-1]

        self.vqvae = ActionVQVAE(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            quantization_cost=quantization_cost,
            entropy_loss_ratio=entropy_loss_ratio,
            entropy_loss_type=entropy_loss_type,
            entropy_temperature=entropy_temperature,
            arch=vqvae_arch,
            action_only_quantization=action_only_quantization,
            reconstruction_loss_type=reconstruction_loss_type,
        )
        self._vqvae_train_state = TrainState.create(
            params=self.vqvae.init(
                next_rng(self.vqvae.rng_keys()),
                jnp.zeros((1, self.observation_dim)),
                jnp.zeros((1, self.action_dim)),
                train=True
            ),
            tx=optax.adam(vqvae_lr),
            apply_fn=None,
        )

        self._vqvae_total_steps = 0
        self.sample_action = sample_action

        self.behavior_policy = FullyConnectedNetwork(
            codebook_size
        )
        behavior_policy_params = self.behavior_policy.init(
            next_rng(self.behavior_policy.rng_keys()),
            jnp.zeros((10, self.observation_dim))
        )
        self._behavior_policy_train_state = TrainState.create(
            params=behavior_policy_params,
            tx=optax.adamw(
                behavior_policy_lr, 
                weight_decay=behavior_policy_weight_decay),
            apply_fn=None
        )
        self._behavior_policy_total_steps = 0

        self._sampler_policy = VQSamplerPolicy(
            self._critic, self.vqvae, self.behavior_policy,
            self._critic.params, self._vqvae_train_state.params, self._behavior_policy_train_state.params, 
            sample_action, A_scaling, kl_divergence_weight
        )
    
    def train_behavior_policy(self, batch):
        self._behavior_policy_train_state, metrics = self._behavior_policy_train_step(
            next_rng(), self._behavior_policy_train_state, batch
        )
        self._behavior_policy_total_steps += 1
        return metrics

    @partial(jax.jit, static_argnames=('self', ))
    def _behavior_policy_train_step(self, rng, train_state, batch):
        observations = batch['observations']
        actions = batch['actions']
        rng_generator = JaxRNG(rng)

        @partial(jax.grad, has_aux=True)
        def grad_fn(train_param, rng):
            observations = batch['observations']
            actions = self.vqvae.apply(
                self._vqvae_train_state.params,
                observations,
                batch['actions'],
                method=self.vqvae.encode
            )

            @wrap_function_with_rng(rng_generator())
            def forward_behavior_policy(rng, *args, **kwargs):
                return self.behavior_policy.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.behavior_policy.rng_keys())
                )
            
            q_values = forward_behavior_policy(train_param, observations)
            log_probs = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(q_values, actions))
            policy_loss = log_probs

            return policy_loss, locals()
        grads, aux_values = grad_fn(train_state.params, rng)
        new_train_state = train_state.apply_gradients(grads=grads)
        metrics = collect_jax_metrics(
            aux_values,
            ['policy_loss', 'log_probs'],
        )
        return new_train_state, metrics
    

    @partial(jax.jit, static_argnames=('self', ))
    def _vqvae_train_step(self, rng, train_state, batch):
        observations = batch['observations']
        actions = batch['actions']
        rng_generator = JaxRNG(rng)

        @partial(jax.grad, has_aux=True)
        def grad_fn(train_params):
            reconstructed, result_dict = self.vqvae.apply(
                train_params,
                observations,
                actions,
                train=True,
            )
            return result_dict['loss'], result_dict

        grads, aux_values = grad_fn(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        metrics = collect_jax_metrics(
            aux_values,
            ['loss', 'reconstruction_loss', 'quantizer_loss', 'e_latent_loss', 'q_latent_loss',
             'entropy_loss', 'action_prior_loss', 'action_prior_accuracy'],
        )
        return new_train_state, metrics

    def train_vqvae(self, batch):
        self._vqvae_train_state, metrics = self._vqvae_train_step(
            next_rng(), self._vqvae_train_state, batch
        )
        self._vqvae_total_steps += 1
        return metrics

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_critic, new_target_critic, new_value, info = self._update_jit(
            self._rng, self._critic, self._target_critic_params, self._vqvae_train_state,
            self._value, batch, self.discount, self.tau, self.expectile,
            self.A_scaling, self.critic_reduction, self._behavior_policy_train_state)

        self._rng = new_rng
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        return info

    @partial(jax.jit, static_argnames=('self', 'critic_reduction'))
    def _update_jit(
        self, rng: PRNGKey, critic: TrainState,
        target_critic_params: Params, vqvae: TrainState, value: TrainState, batch: TrainState,
        discount: float, tau: float, expectile: float, A_scaling: float,
        critic_reduction: str, policy: TrainState
    ) -> Tuple[PRNGKey, TrainState, Params, TrainState, Dict[str,
                                                            float]]:
        observations = batch['observations']
        original_actions = batch['actions']

        actions = self.vqvae.apply(
                vqvae.params,
                observations,
                original_actions,
                method=self.vqvae.encode
            )
        

        log_prob = self.behavior_policy.apply(self._behavior_policy_train_state.params, batch['observations'])

        target_critic = critic.replace(params=target_critic_params)
        new_value, value_info = update_v(target_critic, value, batch, actions, expectile,
                                        critic_reduction)
        key, rng = jax.random.split(rng)

        new_critic, critic_info = update_q(critic, new_value, batch, actions, discount, log_prob, self.kl_divergence_weight)

        new_target_critic_params = soft_target_update(new_critic.params,
                                                    target_critic_params, tau)

        return rng, new_critic, new_target_critic_params, new_value, {
            **critic_info,
            **value_info,
        }

    def get_sampler_policy(self):
        return self._sampler_policy.update_params(
            self._critic.params, self._vqvae_train_state.params, self._behavior_policy_train_state.params
        )