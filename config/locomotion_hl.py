import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ## value kwargs
    ("discount", "d"),
    ("jump", "J"),
    ("scale", "S"),
    ("n_guide_steps", "n"),
    ("t_stopgrad", "ts"),
]

logbase = "logs"

base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 36,
        "jump": 4,
        "jump_action": False,
        "n_diffusion_steps": 20,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 2, 4, 8),
        "dk": 3,
        "ds": 2,
        "dp": 1,
        "uk": 3,
        "us": 2,
        "up": 1,
        "attention": False,
        "kernel_size": 5,
        "dim": 32,
        "attention": False,
        "renderer": "utils.MuJoCoRenderer",
        ## dataset
        "loader": "datasets.SequenceDataset",
        "normalizer": "GaussianNormalizer",
        "preprocess_fns": [],
        "clip_denoised": False,
        "use_padding": True,
        "max_path_length": 1000,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/defaults",
        "exp_name": watch(args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 1e6,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 20000,
        "sample_freq": 20000,
        "n_saves": 5,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
    },
    "values": {
        "model": "models.ValueFunction",
        "diffusion": "models.ValueDiffusion",
        "horizon": 36,
        "jump": 4,
        "jump_action": False,
        "n_diffusion_steps": 20,
        "dim_mults": (1, 2, 4, 8),
        "dk": 3,
        "ds": 2,
        "dp": 1,
        "kernel_size": 5,
        "dim": 32,
        "attention": False,
        "renderer": "utils.MuJoCoRenderer",
        ## value-specific kwargs
        "discount": 0.997,
        "termination_penalty": -100,
        "normed": False,
        ## dataset
        "loader": "datasets.ValueDataset",
        "normalizer": "GaussianNormalizer",
        "preprocess_fns": [],
        "use_padding": True,
        "max_path_length": 1000,
        ## serialization
        "logbase": logbase,
        "prefix": "values/defaults",
        "exp_name": watch(args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "value_l2",
        "n_train_steps": 400e3,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 1000,
        "sample_freq": 0,
        "n_saves": 5,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
    },
    "plan": {
        "guide": "sampling.ValueGuide",
        "policy": "sampling.GuidedPolicy",
        "max_episode_length": 1000,
        "batch_size": 64,
        "preprocess_fns": [],
        "device": "cuda",
        "seed": None,
        ## sample_kwargs
        "n_guide_steps": 2,
        "scale": 0.1,
        "t_stopgrad": 2,
        "scale_grad_by_std": True,
        ## serialization
        "loadbase": None,
        "logbase": logbase,
        "prefix": "plans/",
        "exp_name": watch(args_to_watch),
        "vis_freq": 100,
        "max_render": 8,
        ## diffusion model
        "horizon": 36,
        "jump": 4,
        "jump_action": False,
        "n_diffusion_steps": 20,
        ## value function
        "discount": 0.997,
        ## loading
        "diffusion_loadpath": "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}_J{jump}",
        "value_loadpath": "f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}_J{jump}",
        "diffusion_epoch": "latest",
        "value_epoch": "latest",
        "verbose": True,
        "suffix": "0",
    },
}


# ------------------------ overrides ------------------------#


halfcheetah_medium_expert_v2 = {
    "plan": {
        "scale": 0.001,
        "t_stopgrad": 4,
        "n_guide_steps": 1,
    }
}

halfcheetah_medium_v2 = {
    "plan": {
        "scale": 0.001,
        "t_stopgrad": 4,
        "n_guide_steps": 1,
    }
}

halfcheetah_medium_replay_v2 = {
    "plan": {
        "scale": 0.001,
        "t_stopgrad": 4,
        "n_guide_steps": 1,
    }
}
hopper_medium_expert_v2 = {
    "plan": {
        "scale": 0.0001,
        "t_stopgrad": 2,
        "n_guide_steps": 1,
    }
}

hopper_medium_v2 = {
    "plan": {
        "scale": 0.0001,
        "t_stopgrad": 2,
        "n_guide_steps": 1,
    }
}

hopper_medium_replay_v2 = {
    "plan": {
        "scale": 0.0001,
        "t_stopgrad": 4,
        "n_guide_steps": 1,
    }
}
walker2d_medium_expert_v2 = {
    "plan": {
        "scale": 0.0001,
        "t_stopgrad": 2,
        "n_guide_steps": 1,
    }
}

walker2d_medium_v2 = {
    "plan": {
        "scale": 0.0001,
        "t_stopgrad": 2,
        "n_guide_steps": 1,
    }
}

walker2d_medium_replay_v2 = {
    "plan": {
        "scale": 0.1,
        "t_stopgrad": 4,
        "n_guide_steps": 2,
    }
}
