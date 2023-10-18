from typing import Callable, Optional, Any
from functools import partial
from copy import deepcopy

from ml_collections import ConfigDict
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from .jax_utils import (
    next_rng, value_and_multi_grad, mse_loss, JaxRNG, wrap_function_with_rng,
    collect_jax_metrics
)
from .model import FullyConnectedNetwork, update_target_network
from .utils import prefix_metrics


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

    def __init__(self, qf, vqvae, qf_params, vqvae_params):
        self.qf = qf
        self.vqvae = vqvae
        self.qf_params = qf_params
        self.vqvae_params = vqvae_params

    def update_params(self, qf_params, vqvae_params):
        self.qf_params = qf_params
        self.vqvae_params = vqvae_params
        return self

    @partial(jax.jit, static_argnames=('self',))
    def act(self, rng, qf_params, vqvae_params, observations):
        q_vals = self.qf.apply(qf_params, observations)
        actions = jnp.argmax(q_vals, axis=-1)
        return self.vqvae.apply(
            vqvae_params, observations, actions,
            method=self.vqvae.decode
        )


    def __call__(self, observations, deterministic):
        del deterministic
        actions = self.act(
            next_rng(), self.qf_params, self.vqvae_params, observations
        )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class DQNTrainState(TrainState):
    target_params: Any


class VQN(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.embedding_dim = 128
        config.codebook_size = 64
        config.commitment_cost = 1.0
        config.quantization_cost = 1.0
        config.entropy_loss_ratio = 0.0
        config.entropy_loss_type = "softmax"
        config.entropy_temperature = 1.0
        config.vqvae_arch = '512-512'
        config.action_only_quantization = False
        config.reconstruction_loss_type = 'l2'
        config.vqvae_lr = 3e-4

        config.discount = 0.99
        config.qf_arch = '512-512'
        config.qf_lr = 3e-4
        config.target_update_period = 200
        config.reset_qf = False
        config.td_loss_weight = 1.0

        config.bc_loss_weight = 0.0

        config.action_selection_threshold = 0.0

        config.cql_temp = 1.0
        config.cql_min_q_weight = 0.0
        
        config.qf_weight_decay = 0.0

        config.q_value_penalty_weight = 0.0
        config.q_value_penalty_type = 'l1'
        config.q_value_penalty_aggregation = 'mean'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, observation_dim, action_dim):
        self.config = self.get_default_config(config)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.vqvae = ActionVQVAE(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            embedding_dim=self.config.embedding_dim,
            codebook_size=self.config.codebook_size,
            commitment_cost=self.config.commitment_cost,
            quantization_cost=self.config.quantization_cost,
            entropy_loss_ratio=self.config.entropy_loss_ratio,
            entropy_loss_type=self.config.entropy_loss_type,
            entropy_temperature=self.config.entropy_temperature,
            arch=self.config.vqvae_arch,
            action_only_quantization=self.config.action_only_quantization,
            reconstruction_loss_type=self.config.reconstruction_loss_type,
        )

        self._vqvae_train_state = TrainState.create(
            params=self.vqvae.init(
                next_rng(self.vqvae.rng_keys()),
                jnp.zeros((1, observation_dim)),
                jnp.zeros((1, action_dim)),
                train=True
            ),
            tx=optax.adam(self.config.vqvae_lr),
            apply_fn=None,
        )
        self._vqvae_total_steps = 0

        self.qf = FullyConnectedNetwork(
            output_dim=self.config.codebook_size,
            arch=self.config.qf_arch,
        )

        qf_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((1, observation_dim)),
        )

        self._qf_optimizer = optax.adam(self.config.qf_lr)
        self._qf_train_state = DQNTrainState.create(
            params=qf_params,
            target_params=deepcopy(qf_params),
            tx=optax.adamw(self.config.qf_lr, self.config.qf_weight_decay),
            apply_fn=None,
        )
        self._dqn_total_steps = 0

        self._sampler_policy = VQSamplerPolicy(
            self.qf, self.vqvae,
            self._qf_train_state.params, self._vqvae_train_state.params
        )


    def train_vqvae(self, batch):
        self._vqvae_train_state, metrics = self._vqvae_train_step(
            next_rng(), self._vqvae_train_state, batch
        )
        self._vqvae_total_steps += 1
        return metrics

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

    def train_dqn(self, batch, bc=False):
        self._qf_train_state, metrics = self._dqn_train_step(
            next_rng(), self._qf_train_state, self._vqvae_train_state, batch,
            bc
        )
        self._dqn_total_steps += 1
        return metrics

    @partial(jax.jit, static_argnames=('self', 'bc'))
    def _dqn_train_step(self, rng, qf_train_state, vqvae_train_state, batch, bc=False):
        observations = batch['observations']
        original_actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        rng_generator = JaxRNG(rng)

        actions = self.vqvae.apply(
            vqvae_train_state.params,
            observations,
            original_actions,
            method=self.vqvae.encode
        )

        @partial(jax.grad, has_aux=True)
        def grad_fn(train_params):
            def select_by_action(q_vals, actions):
                return jnp.squeeze(
                    jnp.take_along_axis(
                        q_vals, jnp.expand_dims(actions, -1), axis=-1
                    ),
                    axis=-1
                )

            def select_actions(params, observations):
                q_values = self.qf.apply(params, observations)
                action_priors = jax.nn.softmax(
                    self.vqvae.apply(
                        vqvae_train_state.params,
                        observations,
                        method=self.vqvae.action_prior_logits
                    ),
                    axis=-1
                )
                action_selection_threshold = jnp.minimum(
                    jnp.amax(action_priors, axis=-1, keepdims=True),
                    self.config.action_selection_threshold
                )
                action_mask = (
                    action_priors >= action_selection_threshold
                ).astype(jnp.float32)
                masked_q_values = (
                    action_mask * q_values + (1.0 - action_mask) * jnp.min(q_values)
                )
                return jnp.argmax(masked_q_values, axis=-1)


            q_values = self.qf.apply(train_params, observations)
            current_actions_q_values = select_by_action(q_values, actions)
            next_q_values = self.qf.apply(qf_train_state.target_params, next_observations)
            next_actions = select_actions(train_params, next_observations)
            target_q_values = select_by_action(next_q_values, next_actions)

            td_target = rewards + (1. - dones) * self.config.discount * target_q_values

            td_loss = mse_loss(current_actions_q_values, jax.lax.stop_gradient(td_target))
            loss = self.config.td_loss_weight * td_loss

            current_actions = jnp.argmax(q_values, axis=-1)
            max_q_values = jnp.max(q_values, axis=-1)
            advantage = max_q_values - current_actions_q_values

            policy_dataset_aggrement_rate = jnp.mean(current_actions == actions)
            reconstructed_current_actions = self.vqvae.apply(
                vqvae_train_state.params,
                observations,
                current_actions,
                method=self.vqvae.decode
            )
            current_action_mse = jnp.sum(
                jnp.square(reconstructed_current_actions - original_actions),
                axis=-1
            ).mean()

            bc_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(q_values, actions))
            loss = loss + self.config.bc_loss_weight * bc_loss

            cql_lse_q_values = self.config.cql_temp * jax.scipy.special.logsumexp(
                q_values / self.config.cql_temp, axis=-1
            )
            cql_min_q_loss = jnp.mean(cql_lse_q_values - current_actions_q_values)
            loss = loss + self.config.cql_min_q_weight * cql_min_q_loss

            if self.config.q_value_penalty_aggregation == 'none':
                aggregated_q_values = q_values
            elif self.config.q_value_penalty_aggregation == 'mean':
                aggregated_q_values = jnp.mean(q_values)
            else:
                raise ValueError('Unsupport value penalty aggregation type!')

            if self.config.q_value_penalty_type == 'l1':
                q_value_penalty_loss = jnp.mean(jnp.abs(aggregated_q_values))
            elif self.config.q_value_penalty_type == 'l2':
                q_value_penalty_loss = jnp.mean(jnp.square(aggregated_q_values))
            else:
                raise ValueError('Unsupport value penalty type!')

            loss = loss + self.config.q_value_penalty_weight * q_value_penalty_loss

            if bc:
                loss = bc_loss

            return loss, locals()

        grads, aux_values = grad_fn(qf_train_state.params)
        new_target_params = jax.lax.cond(
            qf_train_state.step % self.config.target_update_period == self.config.target_update_period - 1,
            lambda: qf_train_state.params,
            lambda: qf_train_state.target_params,
        )
        if self.config.reset_qf:
            def reset_qf_params():
                qf_params = self.qf.init(
                    rng_generator(self.qf.rng_keys()),
                    jnp.zeros((1, self.observation_dim)),
                )
                return DQNTrainState.create(
                    params=qf_params,
                    target_params=new_target_params,
                    tx=self._qf_optimizer,
                    apply_fn=None,
                )

            new_qf_train_state = jax.lax.cond(
                qf_train_state.step % self.config.target_update_period == self.config.target_update_period - 1,
                reset_qf_params,
                lambda: qf_train_state.apply_gradients(grads=grads, target_params=new_target_params)
            )
        else:
            new_qf_train_state = qf_train_state.apply_gradients(
                grads=grads, target_params=new_target_params
            )

        metrics = collect_jax_metrics(
            aux_values,
            ['loss', 'current_actions_q_values', 'max_q_values', 'target_q_values',
             'advantage', 'td_target', 'td_loss', 'cql_lse_q_values', 'cql_min_q_loss',
             'policy_dataset_aggrement_rate', 'bc_loss', 'current_action_mse',
             'q_value_penalty_loss'],
        )

        return new_qf_train_state, metrics

    def get_sampler_policy(self):
        return self._sampler_policy.update_params(
            self._qf_train_state.params, self._vqvae_train_state.params
        )

