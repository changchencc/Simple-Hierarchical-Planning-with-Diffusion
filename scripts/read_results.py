import os
import glob
import numpy as np
import json
import pdb

import diffuser.utils as utils


DATASETS = [
    f"{env}-{buffer}-v2"
    for env in ["hopper", "walker2d", "halfcheetah"]
    for buffer in ["medium-replay", "medium", "medium-expert"]
]

LOGBASE = "logs"
TRIAL = "*"
EXP_NAME = "plans*/"
verbose = False


def load_results(paths):
    """
    paths : path to directory containing experiment trials
    """
    scores = []
    for i, path in enumerate(sorted(paths)):
        score = load_result(path)
        if verbose:
            print(path, score)
        if score is None:
            # print(f'Skipping {path}')
            continue
        scores.append(score)

        suffix = path.split("/")[-1]
        # print(suffix, path, score)

    if len(scores) > 0:
        if len(scores) > 5:
            scores = np.stack(
                [np.array(scores)[idx : idx + 5] for idx in range(len(scores) - 5)]
            )
            mean = scores.mean(-1).max()
            idx = scores.mean(-1).argmax()
            scores = scores[idx]
        else:
            mean = np.mean(scores)
    else:
        mean = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    return mean, err, scores


def load_result(path):
    """
    path : path to experiment directory; expects `rollout.json` to be in directory
    """
    try:
        trial = int(path.split("/")[-1])
    except:
        return None
        # pdb.set_trace()
    # if trial >= 300:
    #   return None

    fullpath = os.path.join(path, "rollout.json")

    if not os.path.exists(fullpath):
        return None

    try:
        results = json.load(open(fullpath, "rb"))
    except:
        pdb.set_trace()
    score = results["score"]
    return score * 100


#######################
######## setup ########
#######################

if __name__ == "__main__":

    class Parser(utils.Parser):
        dataset: str = "walker2d-medium-replay-v2"

    args = Parser().parse_args()

    for dataset in [args.dataset] if args.dataset else DATASETS:
        for s in [0.1, 0.01, 0.001, 0.0001]:
            for n_guide_step in [1, 2]:
                for t_stopgrad in [2, 4]:
                    exp_name = f"plans*/H32_T20_d0.997_J4_k5_S{s}_n{n_guide_step}_ts{t_stopgrad}*"

                    subdirs = sorted(
                        glob.glob(os.path.join(LOGBASE, dataset, exp_name))
                    )

                    for subdir in subdirs:
                        reldir = subdir.split("/")[-1]
                        paths = glob.glob(os.path.join(subdir, TRIAL))
                        paths = sorted(paths)

                        mean, err, scores = load_results(paths)
                        if np.isnan(mean):
                            continue
                        path, name = os.path.split(subdir)
                        print(
                            f"{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}"
                        )
                        if verbose:
                            print(scores)
