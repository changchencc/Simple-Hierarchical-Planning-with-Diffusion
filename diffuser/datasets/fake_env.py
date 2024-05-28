import gym
import collections

class FakeEnv:
  def __init__(self, env_name):
    self.name = env_name
    self.env = gym.make(env_name)

  def get_dataset(self):
    return self.env.get_dataset()

  def reset(self, dataset, preprocess_fn):
    self._step = 0
    self.dataset = preprocess_fn(dataset)
    self.N = self.dataset['rewards'].shape[0]

  def step(self, ):
    if self._step >= (self.N - 1):
        return None

    idx = min(self._step, self.N)
    
    reward = self.dataset['rewards'][idx]
    obs = self.dataset['observations'][idx]
    action = self.dataset['actions'][idx]
    terminal = self.dataset['terminals'][idx]
    timeouts = self.dataset['timeouts'][idx]
    info = dict()
    info['timeouts'] = timeouts
    info['actions'] = action
    info['infos/qpos'] = self.dataset['infos/qpos'][idx]
    info['infos/qvel'] = self.dataset['infos/qvel'][idx]
    if 'infos/goal' in self.dataset:
      info['infos/goal'] = self.dataset['infos/goal'][idx]
    if 'next_observations' in self.dataset:
      info['next_observations'] = self.dataset['next_observation'][idx]

    self._step += 1

    return obs, reward, terminal, info