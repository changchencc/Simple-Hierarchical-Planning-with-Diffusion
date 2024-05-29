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
