import os

# os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["MUJOCO_GL"] = "glfw"

os.environ["OPENBLAS_NUM_THREADS"] = "5"

import argparse
from datetime import datetime

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import wandb

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos


def main(config_file):
    config = load_config(config_file)

    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
        get_base_directory() + "/runs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + config.operation.log_dir
    )

    if config.operation.wandb:
        wandb.init(
            project=config.algorithm,
            config=config,
            dir=log_dir,
            sync_tensorboard=True,
        )

    writer = SummaryWriter(log_dir)
    device = config.operation.device

    if config.algorithm == "dreamer-v1":
        agent = Dreamer(obs_shape, discrete_action_bool, action_size, writer, device, config, log_dir)
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(obs_shape, discrete_action_bool, action_size, writer, device, config, log_dir)

    agent.train(env)

    writer.close()
    if config.operation.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dmc-walker-walk.yml",
        help="config file to run(default: dmc-walker-walk.yml)",
    )
    main(parser.parse_args().config)
