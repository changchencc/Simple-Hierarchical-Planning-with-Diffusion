import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.policies import HlGuidedPolicy


# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#


def load_model(args, ll=False):

    # load diffusion model and value function from disk
    diffusion_experiment = utils.load_diffusion(
        args.loadbase,
        args.dataset,
        args.diffusion_loadpath,
        epoch=args.diffusion_epoch,
        seed=args.seed,
    )
    value_experiment = utils.load_diffusion(
        args.loadbase,
        args.dataset,
        args.value_loadpath,
        epoch=args.value_epoch,
        seed=args.seed,
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(diffusion_experiment, value_experiment)

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    ## initialize value guide
    value_function = value_experiment.ema
    if ll:
        value_function.cond_key = [0, args.horizon - 1]
        diffusion.cond_key = [0, args.horizon - 1]
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()

    logger_config = utils.Config(
        utils.Logger,
        renderer=renderer,
        logpath=args.savepath,
        vis_freq=args.vis_freq,
        max_render=args.max_render,
    )

    ## policies are wrappers around an unconditional diffusion model and a value guide
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        jump=args.jump,
        jump_action=args.jump_action,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )

    logger = logger_config()
    policy = policy_config()

    return dataset, policy, diffusion_experiment, value_experiment, logger


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#
class HLParser(utils.Parser):
    dataset: str = "hopper-medium-expert-v2"
    config: str = "config.locomotion_hl"


hlargs = HLParser().parse_args("plan")


class LLParser(utils.Parser):
    dataset: str = "hopper-medium-expert-v2"
    config: str = "config.locomotion_ll"


llargs = LLParser().parse_args("plan")
llargs.suffix = hlargs.suffix
llargs.n_guide_steps = hlargs.n_guide_steps
llargs.scale = hlargs.scale
llargs.t_stopgrad = hlargs.t_stopgrad
llargs.dataset = hlargs.dataset

_, hl_policy, hl_diffusion_experiment, hl_value_experiment, hl_logger = load_model(
    hlargs
)
dataset, ll_policy, ll_diffusion_experiment, ll_value_experiment, ll_logger = (
    load_model(llargs, ll=True)
)

policy = HlGuidedPolicy(hl_policy=hl_policy, ll_policy=ll_policy, jump=hlargs.jump)

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#
#
env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
done = False
for t in range(llargs.max_episode_length):

    if done:
        break
    else:

        if t % 10 == 0:
            print(llargs.savepath, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(
            conditions, batch_size=llargs.batch_size, verbose=llargs.verbose
        )

        ## execute action in environment
        next_observation, reward, terminal, infos = env.step(action)

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        if t % 50 == 0:
            print(
                f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | ",
                flush=True,
            )
        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples, state, rollout, jump=1)

        if terminal:
            done = True
            print(
                f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | ",
                flush=True,
            )
            break

        observation = next_observation

## write results to json file at `args.savepath`
# logger.log(0, samples, state, rollout, jump=1 if args.jump_action else args.jump)

ll_logger.finish(
    t, score, total_reward, terminal, ll_diffusion_experiment, ll_value_experiment
)
