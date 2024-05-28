from collections import namedtuple

# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
import numpy as np

# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple("Trajectories", "actions observations")
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')


class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim

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
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, **kwargs)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        if self.action_dim != 0:
            actions = sample[:, :, : self.action_dim]
            action = actions[0, 0]
        else:
            actions = None
            action = None
        ## extract first action

        # if debug:
        act_dim = self.action_dim
        obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories


"""
class Policy:
    def __init__(
        self,
        diffusion_model,
        normalizer,
        jump=1,
        jump_action=False,
        fourier_feature=False,
    ):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.jump = jump
        self.jump_action = jump_action
        self.fourier_feature = fourier_feature

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

    def __call__(
        self, conditions, debug=False, batch_size=1, hd=False, hl=False, **kwargs
    ):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        if hd:
            hl_sample, ll_sample = self.diffusion_model(conditions, **kwargs)
            if hl:
                sample = hl_sample
            else:
                sample = ll_sample
        else:
            sample = self.diffusion_model(conditions, **kwargs)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, : self.action_dim]
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
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        if hd:
            if hl:
                act_dim = self.action_dim * 15
                obs_dim = self.diffusion_model.hl_diffuser.observation_dim
                if self.fourier_feature:
                    obs_dim = obs_dim // 3
            else:
                act_dim = self.action_dim
                obs_dim = self.diffusion_model.hl_diffuser.observation_dim
                if self.fourier_feature:
                    obs_dim = obs_dim // 3
        else:
            act_dim = self.action_dim
            obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
        # else:
        #     return action

    def sample_with_context(
        self,
        target_obs,
        ctxt,
        hl_ctxt_len,
        jump,
        ll_ctxt_len=0,
        debug=False,
        batch_size=1,
        hd=False,
        hl=False,
        **kwargs
    ):
        if len(ctxt) <= jump:
            hl_ctxt = ctxt[-1:]
        elif len(ctxt) <= hl_ctxt_len:
            hl_ctxt = ctxt[::-1][::jump][::-1]
        else:
            hl_ctxt = ctxt[-hl_ctxt_len + 1 :]
            hl_ctxt = hl_ctxt[::jump]

        hl_cond = dict()
        for i in range(len(hl_ctxt)):
            hl_cond[i] = hl_ctxt[i]
        hl_cond[self.diffusion_model.hl_diffuser.horizon - 1] = target_obs

        hl_cond = self._format_conditions(hl_cond, batch_size)

        hl_samples = self.diffusion_model.hl_diffuser(cond=hl_cond, **kwargs)
        hl_state = hl_samples[
            :, :, self.diffusion_model.hl_diffuser.action_dim :
        ]  # B, M, C
        B, _ = hl_state.shape[:2]

        ll_cond = {0: hl_state[:1, len(hl_ctxt) - 1], jump: hl_state[:1, len(hl_ctxt)]}

        ll_samples = self.diffusion_model.ll_diffuser(cond=ll_cond, **kwargs)
        ll_samples_ = ll_samples.reshape(B, 1, jump + 1, -1)
        ll_samples = torch.cat(
            [ll_samples_[:, 0, :1], ll_samples_[:, :, 1:].reshape(B, jump, -1)], dim=1
        )

        ## run reverse diffusion process
        if hl:
            sample = hl_samples
        else:
            sample = ll_samples
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, : self.action_dim]
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
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        if hl:
            act_dim = self.action_dim * 15
            obs_dim = self.diffusion_model.hl_diffuser.observation_dim
            if self.fourier_feature:
                obs_dim = obs_dim // 3
        else:
            act_dim = self.action_dim
            obs_dim = self.diffusion_model.hl_diffuser.observation_dim
            if self.fourier_feature:
                obs_dim = obs_dim // 3
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
"""
