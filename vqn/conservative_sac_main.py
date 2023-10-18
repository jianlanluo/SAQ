import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import jax
import jax.numpy as jnp
import flax

import gym
import d4rl

import absl.app
import absl.flags

from .vqn import VQN
from .conservative_sac import ConservativeSAC
from .replay_buffer import get_d4rl_dataset, subsample_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .robomimic_utils import (
    SequenceDataset, make_dataset, process_robomimic_dataset, D4RLDataset, get_robomimic_env, 
    ENV_TO_HORIZON_MAP, OBS_KEYS
)
from .utils import (
    Timer, define_flags_with_default, set_random_seed, print_flags,
    get_user_flags, prefix_metrics, WandBLogger
)
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    algorithm='cql',
    max_traj_length=200,
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    bc_epochs=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
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

    dataset = make_dataset(dataset, FLAGS.env)

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )
    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = dataset.sample(FLAGS.batch_size)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    FLAGS.eval_n_trajs, deterministic=False
                )
                reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in trajs])
                metrics['reward_bonus_ct'] =  reward_50_ct               
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
