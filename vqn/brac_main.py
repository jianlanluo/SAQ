import os
import time
from copy import deepcopy
import uuid

import numpy as np
from pprint import pprint
from pprint import pprint

import gym

import jax
import jax.numpy as jnp
import flax

import absl.app
import absl.flags

from .brac import SAC
from .replay_buffer import ReplayBuffer, get_d4rl_dataset, subsample_batch
from .replay_buffer import ReplayBuffer, get_d4rl_dataset, subsample_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import (
    Timer, define_flags_with_default, set_random_seed, print_flags,
    get_user_flags, prefix_metrics, WandBLogger
)
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='HalfCheetah-v2',
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    save_model=False,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    n_pi_beta_epochs=5000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
  
    batch_size=256,

    sac=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, use_tanh=True
    )
    behavior_policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, use_tanh=False
    )
    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)
    
    
    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(FLAGS.sac, behavior_policy, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}

    for pi_beta_epoch in range(FLAGS.n_pi_beta_epochs):
        metrics = {'behavior_policy_epoch': pi_beta_epoch}

        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
            metrics.update(prefix_metrics(sac.train_behavior_policy(batch), 'behavior_policy'))
        wandb_logger.log(metrics)
        pprint(metrics)

    # sac.copy_behavior_policy_to_policy()

    for n_epochs in range(FLAGS.n_epochs):
        metrics = {'brac_epochs': n_epochs}
        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
            metrics.update(prefix_metrics(sac.train(batch), 'brac'))
        
        if n_epochs == 0 or (n_epochs + 1) % FLAGS.eval_period == 0:
            trajs = eval_sampler.sample(
                sampler_policy.update_params(sac.train_params['policy']),
                FLAGS.eval_n_trajs, deterministic=False
            )
            #reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in trajs])
            #metrics['reward_bonus_ct'] =  reward_50_ct
            metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
            metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
            metrics['average_normalizd_return'] = np.mean(
                [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
            )

        wandb_logger.log(metrics)
        pprint(metrics)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    absl.app.run(main)
