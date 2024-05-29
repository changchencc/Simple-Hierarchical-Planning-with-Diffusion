import os
import glob
import numpy as np
import json
import pdb
import diffuser.utils as utils

DATASETS = [
    f"{env}-{buffer}-v2"
    for env in ["maze2d"]
    for buffer in ["umaze", "medium", "large"]
]

LOGBASE = "logs"
TRIAL = "*"
EXP_NAME = "plans*/*"
verbose = False


def load_results(paths):
    """
    paths : path to directory containing experiment trials
    """
    scores = []
    returns = []
    for i, path in enumerate(sorted(paths)):
        score, r = load_result(path)
        if verbose:
            print(path, score)
        if score is None:
            # print(f'Skipping {path}')
            continue
        scores.append(score)
        returns.append(r)

        suffix = path.split("/")[-1]
        # print(suffix, path, score)

    num_ = len(scores)
    if len(scores) > 0:
        if len(scores) > 100:
            scores = np.stack(
                [np.array(scores)[idx : idx + 100] for idx in range(num_ - 100)]
            )
            mean = scores.mean(-1).max()
            idx = scores.mean(-1).argmax()
            scores = scores[idx]
            returns = np.stack(
                [np.array(returns)[idx : idx + 100] for idx in range(num_ - 100)]
            )
            returns = returns[idx]
        else:
            mean = np.mean(scores)
            returns = np.array(returns)
    else:
        mean = np.nan
        sus_rate = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    return mean, err, scores, sus_rate


def load_result(path):
    """
    path : path to experiment directory; expects `rollout.json` to be in directory
    """
    # fullpath = os.path.join(path, 'rollout.json')

    if not os.path.exists(path):
        return None

    results = json.load(open(path, "rb"))
    score = results["score"] * 100
    r = np.sum(results["return"])
    return score, r


#######################
######## setup ########
#######################

if __name__ == "__main__":
    configs = ["config.maze2d_hl"]

    for cfg in configs:

        class Parser(utils.Parser):
            dataset: str = "maze2d-medium-v1"
            config: str = cfg

        args = Parser().parse_args("plan")
        epochs = ["latest"]

        for dataset in [args.dataset] if args.dataset else DATASETS:
            subdir = "/" + os.path.join(*args.savepath.split("/")[:-1])

            reldir = subdir.split("/")[-1]
            paths = glob.glob(os.path.join(subdir, TRIAL, f"*_rollout.json"))
            paths = sorted(paths)

            mean, err, scores, sus_rate = load_results(paths)
            if np.isnan(mean):
                continue
            path, name = os.path.split(subdir)
            print(
                f"{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}"
                f"\nsus_rate: {sus_rate * 100:.2f}"
            )
            if verbose:
                print(scores)
                print(sus_rate)
