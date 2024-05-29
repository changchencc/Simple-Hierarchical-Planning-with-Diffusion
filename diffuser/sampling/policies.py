from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np
import time


Trajectories = namedtuple("Trajectories", "actions observations values")


class GuidedPolicy:

    def __init__(
        self,
        guide,
        diffusion_model,
        normalizer,
        preprocess_fns,
        jump,
        jump_action,
        **sample_kwargs
    ):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.jump = jump
        self.jump_action = jump_action

    def __call__(self, conditions, batch_size=1, verbose=False, return_sample=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(
            conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs
        )
        if return_sample:
            return samples

        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        if self.action_dim != 0:
            actions = trajectories[:, :, : self.action_dim]
            shape = actions.shape
            if self.jump_action:
                actions = self.normalizer.unnormalize(
                    actions.reshape(*shape[:-1], 1, -1), "actions"
                )
            else:
                actions = self.normalizer.unnormalize(
                    actions.reshape(*shape[:-1], self.jump, -1), "actions"
                )
            actions = actions.reshape(*shape[:-1], -1)

            ## extract first action
            action = actions[0, 0]
        else:
            actions = None
            action = None

        normed_observations = trajectories[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            "d -> repeat d",
            repeat=batch_size,
        )
        return conditions


class HlGuidedPolicy:
    def __init__(self, hl_policy, ll_policy, jump) -> None:
        self.hl_policy = hl_policy
        self.ll_policy = ll_policy
        self.jump = jump

    def __call__(
        self,
        conditions,
        batch_size=1,
        verbose=True,
    ):

        _, hl_traj = self.hl_policy(
            conditions, batch_size, verbose, return_sample=False
        )
        hl_state = hl_traj.observations

        B, M = hl_state.shape[:2]
        ll_cond_ = np.stack([hl_state[:, :-1], hl_state[:, 1:]], axis=2)
        ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)[0]

        ll_cond = {
            0: ll_cond_[0],
            self.jump: ll_cond_[-1],
        }

        action, trajectories = self.ll_policy(ll_cond, batch_size, verbose)
        return action, trajectories
