from absl import app, flags
import cloudpickle as pickle
import tqdm
from ml_collections import config_flags

import gym
import numpy as np
import d4rl

from robomimic.utils.dataset import SequenceDataset

from .vqiql import VQIQLLearner, IQLSamplerPolicy, split_into_trajectories
from .replay_buffer import get_d4rl_dataset
from .sampler import TrajSampler
from .robomimic_utils import (
    OfflineDataset, process_robomimic_dataset, D4RLDataset, get_robomimic_env,
    ENV_TO_HORIZON_MAP, OBS_KEYS
)
from .utils import (
    Timer, define_flags_with_default, set_random_seed,
    get_user_flags, prefix_metrics, WandBLogger
)

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags_with_default(
    env='pen-human-v1',
    dataset_dir = '',
    seed=42,
    save_model=False,
    zero_reward=False,

    reward_scale=1.0,
    reward_bias=0.0,

    max_traj_length=200,
    eval_n_trajs=10,
    eval_period=10,

    batch_size=256,
    vqvae_n_epochs=500,
    n_epochs=1000,
    n_pi_beta_epochs=2000,
    n_train_step_per_epoch=50,
    tqdm=True,

    embedding_dim=128,
    codebook_size=64,
    commitment_cost=1.0,
    quantization_cost=1.0,
    entropy_loss_ratio=0.0,
    entropy_loss_type="softmax",
    entropy_temperature=1.0,
    vqvae_arch='512-512',
    sample_action=True,

    policy_weight_decay=0.0,
    kl_divergence_weight=0.0,
    qf_weight_decay=0.0,
    qf_arch='256-256',
    iql_expectile=0.8,
    iql_temperature=0.1,
    iql_bc_loss_weight=0.0,

    logging=WandBLogger.get_default_config(),
)


OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
ENV_TO_HORIZON_MAP = {'lift': 400,
                      'can': 400,
                      'square': 400,
                      'transport': 700,
                      'tool_hang': 700}


def normalize(dataset):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_dataset(dataset, env_name, zero_reward=False):
    if zero_reward:
        dataset['reward'] = np.zeros_like(dataset['rewards'])
    if not env_name in ENV_TO_HORIZON_MAP:
        dataset = OfflineDataset(dataset)

    if 'antmaze' in env_name:
        dataset.dataset_dict['rewards'] *= 100
    elif env_name.split('-')[0] in ['hopper', 'halfcheetah', 'walker2d']:
        dataset.normalize_returns(scaling=1000)

    return dataset

def main(_):

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
        example_ob = dataset.dataset_dict['observations'][0]
        example_action = dataset.dataset_dict['actions'][0]
        env = get_robomimic_env(dataset_path, example_action, FLAGS.env)
        max_len = ENV_TO_HORIZON_MAP[FLAGS.env]
    else:
        env = gym.make(FLAGS.env).unwrapped
        dataset = get_d4rl_dataset(env)
        max_len = FLAGS.max_traj_length
        example_ob = env.observation_space.sample()[np.newaxis]
        example_action = env.action_space.sample()[np.newaxis]


    eval_sampler = TrajSampler(env, max_len)

    hiddens = []
    hidden_str = FLAGS.qf_arch.split("-")
    for i in hidden_str:
        hiddens += [int(i)]

    hiddens = tuple(hiddens)

    agent = VQIQLLearner(FLAGS.seed,
                       example_ob,
                       example_action,
                       embedding_dim=FLAGS.embedding_dim,
                       codebook_size=FLAGS.codebook_size,
                       vqvae_arch=FLAGS.vqvae_arch,
                       expectile=FLAGS.iql_expectile,
                       A_scaling=FLAGS.iql_temperature,
                       policy_weight_decay=FLAGS.policy_weight_decay,
                       decay_steps=None,
                       qf_weight_decay=FLAGS.qf_weight_decay,
                       hidden_dims=hiddens,
                       kl_divergence_weight=FLAGS.kl_divergence_weight,
                       )

    dataset = make_dataset(dataset, FLAGS.env, FLAGS.zero_reward)


    for vqvae_epoch in range(FLAGS.vqvae_n_epochs):
        metrics = {'vqvae_epoch': vqvae_epoch}

        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = dataset.sample(FLAGS.batch_size)
            metrics.update(prefix_metrics(agent.train_vqvae(batch), 'vqvae'))

        wandb_logger.log(metrics)

    for pi_beta_epoch in range(FLAGS.n_pi_beta_epochs):
        metrics = {'behavior_policy_epoch': pi_beta_epoch}

        for batch_idx in range(FLAGS.n_train_step_per_epoch):
            batch = dataset.sample(FLAGS.batch_size)
            metrics.update(prefix_metrics(agent.train_behavior_policy(batch), 'behavior_policy'))
        wandb_logger.log(metrics)


    for i in tqdm.tqdm(range(0, FLAGS.n_epochs),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        metrics = {'steps': i}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = dataset.sample(FLAGS.batch_size)
                update_info = agent.update(batch)
                metrics.update(prefix_metrics(update_info, 'iql'))

        with Timer() as eval_timer:
            if i == 0 or i % FLAGS.eval_period == 0:
                dense_trajs = eval_sampler.sample(agent.get_sampler_policy(), FLAGS.eval_n_trajs, deterministic=False)
                reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in dense_trajs])
                metrics['reward_bonus_ct'] =  reward_50_ct
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in dense_trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in dense_trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in dense_trajs]
                )
                if FLAGS.save_model:
                    save_data = {'iql': agent, 'variant': variant, "epoch": i}
                    wandb_logger.save_pickle(save_data, f'model{i}.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
    
    if FLAGS.save_model:
        save_data = {'iql': agent, 'variant': variant}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    app.run(main)
