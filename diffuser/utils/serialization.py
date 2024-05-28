import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple
from diffuser.models.hier_diffusion import HierDiffusion

DiffusionExperiment = namedtuple(
    "Diffusion", "dataset renderer model diffusion ema trainer epoch"
)
HierDiffusionExperiment = namedtuple(
    "Diffusion", "dataset renderer hl_model ll_model diffusion ema trainer epoch"
)


def mkdir(savepath):
    """
    returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


def get_all_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), "state_*")
    epochs = []
    for state in states:
        epoch = int(state.replace("state_", "").replace(".pt", ""))
        epochs.append(epoch)
    return epochs


def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), "state_*")
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace("state_", "").replace(".pt", ""))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, "rb"))
    print(f"[ utils/serialization ] Loaded config from {loadpath}")
    print(config)
    return config


def load_hl_diffusion(
    *loadpath, jump, hl_epoch="best", ll_epoch="latest", device="cuda:0"
):
    dataset_config = load_config(*loadpath, "dataset_config.pkl")
    render_config = load_config(*loadpath, "render_config.pkl")
    hl_model_config = load_config(*loadpath, "hl_model_config.pkl")
    ll_model_config = load_config(*loadpath, "ll_model_config.pkl")
    hl_diffusion_config = load_config(*loadpath, "hl_diffusion_config.pkl")
    ll_diffusion_config = load_config(*loadpath, "ll_diffusion_config.pkl")
    trainer_config = load_config(*loadpath, "trainer_config.pkl")

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict["results_folder"] = os.path.join(*loadpath)

    dataset = dataset_config()
    val_dataset = dataset_config(split="validation")
    renderer = render_config()
    if "final_k" in hl_model_config:
        hl_model = hl_model_config()
    else:
        if dataset_config.env == "flex-maze":
            hl_model = hl_model_config(final_k=False)
        else:
            hl_model = hl_model_config(final_k=True)
    if "final_k" in ll_model_config:
        ll_model = ll_model_config()
    else:
        if dataset_config.env == "flex-maze":
            ll_model = ll_model_config(final_k=False)
        else:
            ll_model = ll_model_config(final_k=True)
    hl_diffusion = hl_diffusion_config(hl_model)
    ll_diffusion = ll_diffusion_config(ll_model)
    hier_diffusion = HierDiffusion(hl_diffusion, ll_diffusion, dataset.action_dim, jump)
    trainer = trainer_config(hier_diffusion, dataset, val_dataset, renderer)

    if ll_epoch == "latest":
        ll_epoch = get_latest_epoch(loadpath)
    if ll_epoch != -1:
        print(f"\n[ utils/serialization ] Loading ll_model epoch: {ll_epoch}\n")
        trainer.load_ll_diffusion(ll_epoch)

    if hl_epoch == "latest":
        hl_epoch = get_latest_epoch(loadpath)
    if hl_epoch != -1:
        print(f"\n[ utils/serialization ] Loading hl_model epoch: {hl_epoch}\n")
        trainer.load_hl_diffusion(hl_epoch)

    return HierDiffusionExperiment(
        dataset,
        renderer,
        hl_model,
        ll_model,
        hier_diffusion,
        trainer.ema_model,
        trainer,
        trainer.step,
    )


def load_diffusion(*loadpath, epoch="latest", device="cuda:0"):
    dataset_config = load_config(*loadpath, "dataset_config.pkl")
    render_config = load_config(*loadpath, "render_config.pkl")
    model_config = load_config(*loadpath, "model_config.pkl")
    diffusion_config = load_config(*loadpath, "diffusion_config.pkl")
    trainer_config = load_config(*loadpath, "trainer_config.pkl")

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict["results_folder"] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == "latest":
        epoch = get_latest_epoch(loadpath)

    if epoch != -1:
        print(f"\n[ utils/serialization ] Loading model epoch: {epoch}\n")

        trainer.load(epoch)

    return DiffusionExperiment(
        dataset, renderer, model, diffusion, trainer.ema_model, trainer, trainer.step
    )
