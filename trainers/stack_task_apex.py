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
from ray.tune.registry import register_env


def env_creator(env_config):
    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "Sawyer"
    options["controller_configs"] = load_controller_config(default_controller="JOINT_VELOCITY")
    env = GymWrapper(suite.make(
        **options,
        horizon=150,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
    ))
    return env  # return an env instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_workers", type=int, default=16)
    args = parser.parse_args()
    stop_reward = 120
    ray.init()
    register_env("stack_robot", env_creator)

    APEX_DEFAULT_CONFIG = {
        "optimizer": merge_dicts(
            dqn.DEFAULT_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        "n_step": 3,
        "num_gpus": 1,
        "num_workers": args.n_workers,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "rollout_fragment_length": 50,
        "target_network_update_freq": 500000,
        "timesteps_per_iteration": 25000,
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
        # === Evaluation ===
        "evaluation_interval": 500,
        "evaluation_num_episodes": 10,
        "env": 'stack_robot',
        "framework": "torch",
        "model": {"fcnet_hiddens": [400, 300], "fcnet_activation": "tanh", "no_final_linear": False,
                  # Whether layers should be shared for the value function.
                  "vf_share_layers": False, }}

    stop = {
        "timesteps_total": 10000000,
        "episode_reward_mean": stop_reward,
    }

    tune.run("APEX_DDPG", config=APEX_DEFAULT_CONFIG, stop=stop, verbose=1)
    ray.shutdown()

