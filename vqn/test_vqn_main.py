import os
import time
from copy import deepcopy
import uuid

import numpy as np
from pprint import pprint

import jax
import jax.numpy as jnp
import flax

import gym
import d4rl

import absl.app
import absl.flags

from .test_vqn import VQN
from .replay_buffer import get_d4rl_dataset, subsample_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import (
    Timer, define_flags_with_default, set_random_seed, print_flags,
    get_user_flags, prefix_metrics, WandBLogger
)


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=200,
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,


    vqvae_n_epochs=500,
    dqn_n_epochs=1000,
    bc_epochs=1001,
    n_train_step_per_epoch=10,
    eval_period=10,
    eval_n_trajs=5,

    vqn=VQN.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    vqn = VQN(FLAGS.vqn, observation_dim, action_dim)
    import pdb; pdb.set_trace()
    for vqvae_epoch in range(FLAGS.vqvae_n_epochs):
        metrics = {'vqvae_epoch': vqvae_epoch}

        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
            metrics.update(prefix_metrics(vqn.train_vqvae(batch), 'vqvae'))
        
        wandb_logger.log(metrics)
        pprint(metrics)


if __name__ == '__main__':
    absl.app.run(main)
