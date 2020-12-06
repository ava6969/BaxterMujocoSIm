import os

from cloudpickle import cloudpickle
from ray.tune.utils import merge_dicts

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune
import gym

from robosuite.wrappers import GymWrapper
import argparse
import gym, ray

from ray.rllib.agents import ddpg, dqn
from ray.tune.registry import register_env, get_trainable_cls


def eval_env_creator(env_config):
    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "UR5e"
    options["controller_configs"] = load_controller_config(default_controller="JOINT_TORQUE")
    env = GymWrapper(suite.make(
        **options,
        reward_shaping=True,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        horizon=750,
        control_freq = 20,
    ))
    return env  # return an env instance


def env_creator(env_config):
    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "Sawyer"
    options["controller_configs"] = load_controller_config(default_controller="JOINT_TORQUE")
    env = GymWrapper(suite.make(
        **options,
        horizon=750,
        control_freq = 20,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
    ))
    return env  # return an env instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_workers", type=int, default=1)
    parser.add_argument("-e", "--eval", action='store_true')
    parser.add_argument("-c", "--chk_num", type=int)
    args = parser.parse_args()
    ray.init()
    register_env("stack_robot", env_creator)
    register_env("stack_robot_eval", eval_env_creator)
    checkpoint = f'/home/dewe/ray_results/DDPG/DDPG_stack_robot_d3161_00000_0_2020-12-02_14-16-05/checkpoint_{args.chk_num}/checkpoint-{args.chk_num}'
    if args.eval:
        config = {}

        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)

        # Set num_workers to be at least 2.
        if "num_workers" in config:
            config["num_workers"] = 1

        # Make sure worker 0 has an Env.
        config["create_env_on_driver"] = False

        # Create the Trainer from config.
        cls = get_trainable_cls('DDPG')
        agent = cls(config=config)
        # Load state from checkpoint.
        agent.restore(checkpoint)

        # run until episode ends
        episode_reward = 0
        done = False
        eval_env = eval_env_creator(None)
        obs = eval_env.reset()
        for i in range(5 * 500):
            action = agent.compute_action(obs)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            #eval_env.render()
            if done:
                obs = eval_env.reset()
    else:
        config = ddpg.DEFAULT_CONFIG
        config["num_gpus"] = 0.5
        config["num_workers"] = args.n_workers
        config["learning_starts"] =  10000
        config["evaluation_interval"] = 5
        config["evaluation_num_episodes"] = 10
        config["env"] = "stack_robot"
        config["framework"] ="torch"

        stop = {
            "timesteps_total": 10000000,
        }

        tune.run("DDPG", config=config, stop=stop, verbose=1, checkpoint_freq=10)
        ray.shutdown()

