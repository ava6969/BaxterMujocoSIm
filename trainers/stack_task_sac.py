import copy

from ray.tune.utils import merge_dicts

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune, cloudpickle
import gym
import os
from robosuite.wrappers import GymWrapper
import argparse
import gym, ray

from ray.rllib.agents import sac
from ray.tune.registry import register_env, get_trainable_cls


def eval_env_creator(env_config):
    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "Sawyer"
    options["controller_configs"] = load_controller_config(default_controller="JOINT_VELOCITY")
    env = GymWrapper(suite.make(
        **options,
        reward_shaping=True,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
    ))
    return env  # return an env instance
def env_creator(env_config):
    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "Sawyer"
    options["controller_configs"] = load_controller_config(default_controller="JOINT_VELOCITY")
    env = GymWrapper(suite.make(
        **options,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
    ))
    return env  # return an env instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_workers", type=int, default=16)
    parser.add_argument("-e", "--eval", action='store_true')
    parser.add_argument("-c", "--chk_num", type=int)
    args = parser.parse_args()
    stop_reward = 350

    register_env("stack_robot", env_creator)
    register_env("stack_robot_eval", eval_env_creator)
    ray.init()

    config = sac.DEFAULT_CONFIG
    # === Evaluation ===
    config["num_workers"] = args.n_workers

    config["evaluation_interval"] = 500
    config["Q_model"] = {
        "fcnet_activation" : "relu",
        "fcnet_hiddens" : [400, 300]
    }
    # Model options for the policy function.
    config["policy_model"] = {
        "fcnet_activation": "tanh",
        "fcnet_hiddens": [400, 300]
    }
    config["evaluation_num_episodes"] = 10

    config["env"] = "stack_robot"
    config["framework"] = "torch"
    config["num_gpus"] = 1
    if args.eval:
        # ray.init()
        checkpoint = f'/home/dewe/ray_results/PPO/PPO_stack_robot_2149d_00000_0_2020-11-30_17-41-04/checkpoint_{args.chk_num}/checkpoint-{args.chk_num}'
        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        else:
            with open(config_path, "rb") as f:
                config = cloudpickle.load(f)

        # Set num_workers to be at least 2.
        if "num_workers" in config:
            config["num_workers"] = 1

        # Make sure worker 0 has an Env.
        config["create_env_on_driver"] = False

        # Create the Trainer from config.
        cls = get_trainable_cls('PPO')
        agent = cls(config=config)
        # Load state from checkpoint.
        agent.restore(checkpoint)

        # run until episode ends
        episode_reward = 0
        done = False
        eval_env = eval_env_creator(None)
        obs = eval_env.reset()
        for i in range(20 * 500):
            action = agent.compute_action(obs)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            eval_env.render()
            if done:
                obs = eval_env.reset()

    else:
        stop = {
            "timesteps_total": 10000000,
            "episode_reward_mean": stop_reward,
        }

        tune.run("SAC", config=config, stop=stop, verbose=1, checkpoint_freq=50, resume=False)
        ray.shutdown()
