import numpy as np
import collections
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, process_maze2d_episode
from .normalization import DatasetNormalizer
from collections import namedtuple
from .fake_env import FakeEnv
import matplotlib.pyplot as plt
import torch
import pdb


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class ReplayBuffer:
    def __init__(
        self,
        max_n_episodes,
        max_path_length,
        termination_penalty,
    ):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=np.int),
            "priority": np.ones(max_n_episodes, dtype=np.int),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return "[ datasets/buffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self["path_lengths"])

    def _add_keys(self, path):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        can access fields with `buffer.observations`
        instead of `buffer['observations']`
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {
            k: v for k, v in self._dict.items() if k not in ["path_lengths", "priority"]
        }.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        if self._count >= self.max_n_episodes:
            return

        # for antmaze
        obs = path["observations"]

        path_length = len(path["observations"])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path["terminals"].any() and self.termination_penalty is not None:
            assert not path[
                "timeouts"
            ].any(), "Penalized a timeout episode for early termination"
            self._dict["rewards"][
                self._count, path_length - 1
            ] += self.termination_penalty

        ## record path length
        self._dict["path_lengths"][self._count] = path_length
        if "pass_goal" in path:
            if path["pass_goal"].any():
                self._dict["priority"][self._count] = 2

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict["path_lengths"][path_ind]
        new = min(step, old)
        self._dict["path_lengths"][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ["path_lengths"]:
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()
        print(f"[ datasets/buffer ] Finalized replay buffer | {self._count} episodes")


class OnlineReplayBuffer:
    def __init__(
        self,
        env="maze2d-large-v1",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=["maze2d_set_terminals"],
        max_path_length=40000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=False,
        batch_size=32,
        seed=0,
    ):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=np.int),
        }

        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.batch_size = batch_size
        self.seed = seed

        self.data_env = FakeEnv(env_name=env)
        self.env = env = load_environment(env)

        dataset = self.data_env.get_dataset()
        self.normalizer = DatasetNormalizer(dataset, normalizer)
        preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.data_env.reset(dataset, preprocess_fn)

        self.on_going = collections.defaultdict(list)
        self.horizon = horizon
        self.use_padding = use_padding

        self.observation_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]

    def __repr__(self):
        return "[ datasets/buffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self["path_lengths"])

    def _add_keys(self, path):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        can access fields with `buffer.observations`
        instead of `buffer['observations']`
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "path_lengths"}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_step(self, obs, reward, terminal, info):
        self.on_going["rewards"].append(reward)
        self.on_going["observations"].append(obs)
        self.on_going["terminals"].append(terminal)

        done_bool = bool(terminal)
        use_timeouts = "timeouts" in info
        if use_timeouts:
            final_timestep = info["timeouts"]
        else:
            final_timestep = (
                len(self.on_going["rewards"]) == self.env._max_episode_steps - 1
            )

        for k, v in info.items():
            self.on_going[k].append(v)

        if done_bool or final_timestep or (len(self.on_going["rewards"]) >= 1000):
            episode = dict()
            for k, v in self.on_going.items():
                episode[k] = np.array(v)

            if "maze2d" in self.env.name:
                episode = process_maze2d_episode(episode)

            normed_episode = self.normalize(episode)
            for k, v in normed_episode.items():
                episode[k] = v

            self.add_path(episode)
            self.on_going = collections.defaultdict(list)

    def normalize(self, episode, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        normed = dict()
        for key in keys:
            normed["normed_" + key] = self.normalizer(episode[key], key)
        return normed

    def batch_dataset(self, batch_size):
        generator = self.sample_episodes()
        return self.from_generator(generator, batch_size)

    def get_conditions(self, observations):
        if "maze2d" in self.env.name:
            """
            condition on both the current observation and the last observation in the plan
            """
            return {
                0: observations[0],
                self.horizon - 1: observations[-1],
            }

        else:
            """
            condition on current observation for planning
            """
            return {0: observations[0]}

    def sample_episodes(self, balance=False):
        random = np.random.RandomState(self.seed)
        while True:
            n_episode = self._count
            idx = random.choice(np.arange(n_episode))
            total = self._dict["path_lengths"][idx]
            available = total - self.horizon
            if available < 1:
                # print(f'Skipped short episode of length {available}.')
                continue
            if balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available + 1))

            observations = self._dict["normed_observations"][
                idx, index : index + self.horizon
            ]
            actions = self._dict["normed_actions"][idx, index : index + self.horizon]
            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1)

            yield conditions, trajectories

    def from_generator(self, generator, batch_size):
        while True:
            batch_cond_ = collections.defaultdict(list)
            batch_traj = []

            for _ in range(batch_size):
                cond, traj = next(generator)

                batch_traj.append(traj)
                for k, v in cond.items():
                    batch_cond_[k].append(v)

            batch_traj = torch.tensor(np.stack(batch_traj, axis=0), dtype=torch.float32)
            batch_cond = dict()
            for k, v in batch_cond_.items():
                batch_cond[k] = torch.tensor(np.stack(v, axis=0), dtype=torch.float32)
            batch = Batch(batch_traj, batch_cond)

            yield batch

    def add_path(self, path):
        path_length = len(path["observations"])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path["terminals"].any() and self.termination_penalty is not None:
            assert not path[
                "timeouts"
            ].any(), "Penalized a timeout episode for early termination"
            self._dict["rewards"][
                self._count, path_length - 1
            ] += self.termination_penalty

        ## record path length
        self._dict["path_lengths"][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict["path_lengths"][path_ind]
        new = min(step, old)
        self._dict["path_lengths"][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ["path_lengths"]:
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()
        print(f"[ datasets/buffer ] Finalized replay buffer | {self._count} episodes")

    def plot_coverage(self, save_path):
        n_epi = self._count
        xy = []
        path_lengths = self._dict["path_lengths"]
        for i in range(n_epi):
            xy.append(self._dict["observations"][i][: path_lengths[i]])

        xy = np.concatenate(xy)
        plt.scatter(xy[:, 0], xy[:, 1])
        plt.savefig(save_path)
