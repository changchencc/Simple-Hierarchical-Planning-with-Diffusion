import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models.hier_diffusion import HierDiffusion
import os


class HLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d_hl"


hl_args = HLParser().parse_args("plan")


class LLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d_ll"


ll_args = LLParser().parse_args("plan")
# ---------------------------------- setup ----------------------------------#

# ---------------------------------- loading ----------------------------------#


n_samples = 500

loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)


hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
hl_policy = Policy(hl_diffusion, dataset.normalizer)

ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)

env_eval = datasets.load_environment(hl_args.dataset)

target = env_eval._target
hl_cond = {
    hl_diffusion.horizon - 1: np.array([*target, 0, 0]),
}

total_rewards = []
scores = []
rollouts = []
plans = []
track_action = []


for i in range(n_samples):
    observation = env_eval.reset()
    init_obs = observation.copy()
    observation = env_eval._get_obs()
    rollout = [observation.copy()]

    hl_cond[0] = observation
    action, samples = hl_policy(hl_cond, batch_size=hl_args.batch_size)
    hl_plan = samples.observations

    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        ll_args.horizon - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )
    ll_sequence = ll_samples[0]
    total_reward = []
    action_list = []

    max_episode_steps = env_eval.max_episode_steps
    finished = False
    t = 0
    while t < max_episode_steps:
        if finished:
            break
        else:
            if t < len(ll_sequence) - 1:
                next_waypoint = ll_sequence[t]
            else:
                next_waypoint = ll_sequence[-1].copy()
                next_waypoint[2:] = 0

            state = observation.copy()
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = env_eval.step(action)
            t += 1
            total_reward.append(reward)
            score = env_eval.get_normalized_score(sum(total_reward))

            ## update rollout observations
            rollout.append(next_observation.copy())
            if terminal or t >= max_episode_steps:
                finished = True
                print(
                    f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {sum(total_reward):.2f} | score: {score:.4f} | "
                )
                break
            observation = next_observation

    rollouts.append(rollout)
    total_rewards.append(total_reward)
    scores.append(env_eval.get_normalized_score(sum(total_reward)))

    ## save result as a json file
    json_path = join(hl_args.savepath, f"idx{i}_rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
    }
    json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)
