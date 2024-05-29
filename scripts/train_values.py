import diffuser.utils as utils
import pdb
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "walker2d-medium-replay-v2"
    config: str = "config.locomotion_hl"


args = Parser().parse_args("values")


# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    jump=args.jump,
    jump_action=args.jump_action,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
val_dataset = dataset_config(split="validation")
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim * args.jump
if args.jump_action:
    action_dim = action_dim // args.jump
    if args.jump_action == "none":
        action_dim = 0

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

transition_dim = observation_dim + action_dim
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon // args.jump,
    transition_dim=transition_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    kernel_size=args.kernel_size,
    attention=args.attention,
    attn_jump=args.attn_jump,
    dk=args.dk,
    ds=args.ds,
    dp=args.dp,
    dim=args.dim,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon // args.jump,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, val_dataset, renderer)

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.savepath, epoch="latest"
)
model = diffusion_experiment.model
trainer = diffusion_experiment.trainer
diffusion = diffusion_experiment.diffusion
start_epoch = int(diffusion_experiment.epoch // args.n_steps_per_epoch)

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])

loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
train_writer = SummaryWriter(log_dir=args.savepath + "-train")

for i in range(start_epoch, n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
