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

from robomimic.utils.dataset import SequenceDataset

from .vqn import VQN
from .replay_buffer import get_d4rl_dataset, subsample_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .robomimic_utils import (
    make_dataset, process_robomimic_dataset, D4RLDataset, get_robomimic_env, 
    ENV_TO_HORIZON_MAP, OBS_KEYS
)
from .utils import (
    Timer, define_flags_with_default, set_random_seed, print_flags,
    get_user_flags, prefix_metrics, WandBLogger
)


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=200,
    algorithm='vqn',
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

    if FLAGS.env in ENV_TO_HORIZON_MAP:
        dataset_path = f'./robomimic/datasets/{FLAGS.env}/low_dim_v141.hdf5'
        seq_dataset = SequenceDataset(hdf5_path=dataset_path, 
                                    obs_keys=OBS_KEYS, 
                                    dataset_keys=("actions", "rewards", "dones"), 
                                    hdf5_cache_mode="all", 
                                    load_next_obs=True)
        dataset = process_robomimic_dataset(seq_dataset)
        dataset = D4RLDataset(env=None, custom_dataset=dataset)
        example_ob = dataset.dataset_dict['observations'][0][np.newaxis]
        example_action = dataset.dataset_dict['actions'][0][np.newaxis]
        env = get_robomimic_env(dataset_path, example_action, FLAGS.env)
        max_len = ENV_TO_HORIZON_MAP[FLAGS.env]
    else:
        env = gym.make(FLAGS.env).unwrapped
        dataset = get_d4rl_dataset(env)
        dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
        dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

        max_len = FLAGS.max_traj_length
        example_ob = env.observation_space.sample()[np.newaxis]
        example_action = env.action_space.sample()[np.newaxis]


    eval_sampler = TrajSampler(env, max_len)

    observation_dim = example_ob.shape[1]
    action_dim = example_action.shape[1]

    vqn = VQN(FLAGS.vqn, observation_dim, action_dim)

    dataset = make_dataset(dataset, FLAGS.env)

    for vqvae_epoch in range(FLAGS.vqvae_n_epochs):
        metrics = {'vqvae_epoch': vqvae_epoch}

        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = dataset.sample(FLAGS.batch_size)
            metrics.update(prefix_metrics(vqn.train_vqvae(batch), 'vqvae'))
        
        wandb_logger.log(metrics)
        pprint(metrics)

    for dqn_epoch in range(FLAGS.dqn_n_epochs):
        metrics = {'dqn_epoch': dqn_epoch}
        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = dataset.sample(FLAGS.batch_size)
            metrics.update(prefix_metrics(vqn.train_dqn(batch, bc=dqn_epoch < FLAGS.bc_epochs), 'dqn'))

        if dqn_epoch == 0 or (dqn_epoch + 1) % FLAGS.eval_period == 0:
            trajs = eval_sampler.sample(vqn.get_sampler_policy(), FLAGS.eval_n_trajs, deterministic=False)
            reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in trajs])
            metrics['reward_bonus_ct'] =  reward_50_ct

            metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
            metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
            metrics['average_normalizd_return'] = np.mean(
                [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
            )

        wandb_logger.log(metrics)
        pprint(metrics)



if __name__ == '__main__':
    absl.app.run(main)
