from copy import deepcopy
import collections
import d4rl
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
import gym
from gym.utils import seeding
import numpy as np

from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict

from robomimic.utils.dataset import SequenceDataset
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
ENV_TO_HORIZON_MAP = {'lift': 400,
                      'can': 400,
                      'square': 400,
                      'transport': 700,
                      'tool_hang': 700}

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


DataType = Union[np.ndarray, Dict[str, 'DataType']]
DatasetDict = Dict[str, DataType]


def make_dataset(dataset, env_name):
    if not env_name in ENV_TO_HORIZON_MAP:
        dataset = OfflineDataset(dataset)
    return dataset

# Converts Robomimic SequenceDataset into D4RLDataset format
def process_robomimic_dataset(seq_dataset):
    seq_dataset = seq_dataset.getitem_cache

    for i in range(len(seq_dataset)):
        seq_dataset[i]['obs'] = np.concatenate([seq_dataset[i]['obs'][key] 
                                                for key in OBS_KEYS], axis=1)
        seq_dataset[i]['next_obs'] = np.concatenate([seq_dataset[i]['next_obs'][key] 
                                                     for key in OBS_KEYS], axis=1)

    dataset = {'actions': np.concatenate([seq_dataset[i]['actions'] for i in range(len(seq_dataset))]),
               'rewards': np.concatenate([seq_dataset[i]['rewards'] for i in range(len(seq_dataset))]),
               'terminals': np.concatenate([seq_dataset[i]['dones'] for i in range(len(seq_dataset))]),
               'observations': np.concatenate([seq_dataset[i]['obs'] for i in range(len(seq_dataset))]),
               'next_observations': np.concatenate([seq_dataset[i]['next_obs'] for i in range(len(seq_dataset))])}
    return dataset

def get_robomimic_env(dataset_path, example_action, env_name):
    # Initialize ObsUtils environment variables
    ObsUtils.initialize_obs_utils_with_config(config_factory(algo_name='iql'))
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False, 
        use_image_obs=False,
    )
    env = RobosuiteGymWrapper(env, ENV_TO_HORIZON_MAP[env_name], example_action)
    return env

class RobosuiteGymWrapper():
    def __init__(self, env, horizon, example_action):
        self.env = env
        self.horizon = horizon
        self.timestep = 0
        self.returns = 0
        # Hack as robosuite environment does not have action_space attribute
        self.action_space = example_action 

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        next_obs = deepcopy(next_obs)
        next_obs = self._process_obs(next_obs)
        success = self.env.is_success()["task"]
        state_dict = self.env.get_state()

        self.timestep += 1
        self.returns += reward
        terminated = done or success or self.timestep >= self.horizon
        info = {'episode': {"return": self.returns, "length": self.timestep}} if terminated else None
        return next_obs, reward, terminated, info

    def reset(self):
        obs = self.env.reset()
        state_dict = self.env.get_state()
        # Hack that is necessary for robosuite tasks for deterministic action playback
        obs = self.env.reset_to(state_dict)
        obs = self._process_obs(obs)
        self.timestep = 0
        self.returns = 0
        return obs

    def render(self):
        return None # TODO

    def get_normalized_score(self, rewards):
        return rewards

    def _process_obs(self, obs):
        new_obs = np.concatenate([obs[k] for k in OBS_KEYS], axis=-1)
        return new_obs


def _check_lengths(dataset_dict: DatasetDict,
                   dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, 'Inconsistent item lengths in the dataset.'
        else:
            raise TypeError('Unsupported type.')
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError('Unsupported type.')
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict],
            indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def split(self, ratio: float) -> Tuple['Dataset', 'Dataset']:
        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[:int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio):]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[:int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio):]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict['rewards'][i]

            if self.dataset_dict['dones'][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(self,
               percentile: Optional[float] = None,
               threshold: Optional[float] = None):
        assert ((percentile is None and threshold is not None)
                or (percentile is not None and threshold is None))

        (episode_starts, episode_ends,
         episode_returns) = self._trajectory_boundaries_and_returns()

        if percentile is not None:
            threshold = np.percentile(episode_returns, 100 - percentile)

        bool_indx = np.full((len(self), ), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i]:episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict['rewards'] /= (np.max(episode_returns) -
                                         np.min(episode_returns))
        self.dataset_dict['rewards'] *= scaling


class OfflineDataset(Dataset):

    def __init__(self,
                 dataset_dict: dict,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict['actions'] = np.clip(dataset_dict['actions'], -lim,
                                              lim)

        dones = np.full_like(dataset_dict['rewards'], False, dtype=bool)

        for i in range(len(dones) - 1):
            if np.linalg.norm(dataset_dict['observations'][i + 1] -
                              dataset_dict['next_observations'][i]
                              ) > 1e-6 or dataset_dict['dones'][i] == 1.0:
                dones[i] = True

        dones[-1] = True

        dataset_dict['masks'] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict['dones'] = dones

        super().__init__(dataset_dict)


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 ignore_done: bool = False,
                 custom_dataset: dict = None):
        if custom_dataset:
            if env is not None:
                dataset = d4rl.qlearning_dataset(env, dataset=custom_dataset)
            else:
                dataset = custom_dataset
            print("Loaded custom dataset")
        else:
            dataset = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if ignore_done:
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            else:
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
        dones_float[-1] = 1
        dataset_dict = {
                'observations': dataset['observations'].astype(np.float32),
                'actions': dataset['actions'].astype(np.float32),
                'rewards': dataset['rewards'].astype(np.float32),
                'masks': 1.0 - dataset['terminals'].astype(np.float32),
                'dones': dones_float.astype(np.float32),
                'next_observations': dataset['next_observations'].astype(
                    np.float32)
            }
        super().__init__(dataset_dict)
