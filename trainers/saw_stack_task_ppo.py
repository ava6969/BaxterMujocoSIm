import copy
import json
import time

import imageio
from matplotlib.pyplot import step
from ray.tune.utils import merge_dicts
import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune, cloudpickle
from gym.wrappers import Monitor
import os
from robosuite.wrappers import GymWrapper
import argparse
import gym, ray
import glob
from ray.rllib.agents import ppo
from ray.tune.registry import register_env, get_trainable_cls
from robosuite.wrappers import DemoSamplerWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def eval_env_creator(env_config):
    return CombinedEnv(True)

def env_creator(env_config):
   return CombinedEnv(False)


def record_video(agent, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """

  done = False
  eval_env = DummyVecEnv([lambda: env_creator({}) ])
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()

  for i in range(video_length):
      action = agent.compute_action(obs.squeeze(0))
      action = np.expand_dims(action, 0)
      obs, _, _, _ = eval_env.step(action)

  eval_env.close()


class CombinedEnv(gym.Env):
    def __init__(self, render, ):
        options = {}
        options["env_name"] = "Stack"
        options["robots"] = "Sawyer"
        options["controller_configs"] = load_controller_config(default_controller="JOINT_VELOCITY")
        env = suite.make(
            **options,
            reward_shaping=False,
            has_renderer=render,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            horizon=500,
            control_freq=25,
        )

        self.eval = render
        self.gym_env = GymWrapper(env)
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self.action_spec = self.gym_env.action_spec
        self.action_dim = self.gym_env.action_dim

    def step(self, action):
        return self.gym_env.step(action)

    def reset(self):
        # if self.eval:
        #     return self.gym_env.reset()
        return self.gym_env.reset()

    def render(self, mode='human'):
        self.gym_env.render()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--name", type=str)
    parser.add_argument("-n", "--n_workers", type=int, default=16)
    parser.add_argument("-e", "--eval", action='store_true')
    parser.add_argument("-c", "--chk_num", type=int)
    parser.add_argument("--skip_frame", type=int, default=0)
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("-r", "--record", action='store_true')
    parser.add_argument("--camera", type=str, default="agentview", help="Name of camera to render")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    register_env("stack_robot", env_creator)
    register_env("stack_robot_eval", eval_env_creator)
    ray.init()

    config = ppo.DEFAULT_CONFIG
    # === Evaluation ===
    config["num_workers"] = args.n_workers
    config["train_batch_size"] = 153760
    config["sgd_minibatch_size"] = 4960
    # Coefficient of the entropy regularizer.
    # Decay schedule for the entropy regularizer.
    config["env"] = "stack_robot"
    config["framework"] = "torch"
    config["lambda"] = 0.95
    config["gamma"] = 0.998
    config["entropy_coeff"] = 0.01
    config["vf_loss_coeff"] = 0.5
    config["clip_param"] = 0.2
    config["num_gpus"] = 0.5
    config["grad_clip"] = 0.5
    config["lr"] = .0003

    config["model"] = { "fcnet_hiddens": [400, 300], "fcnet_activation": "relu", "vf_share_layers": False}

    dir = f'/home/dewe/ray_results/{args.name}/*/'
    dirs = glob.glob(dir)
    checkpoint=None
    if len(dirs):
        checkpoint = f'{dirs[0]}checkpoint_{args.chk_num}/checkpoint-{args.chk_num}'
    if args.eval:
        # ray.init()
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

        # print(agent.get_policy().action_space)
        # print(agent.get_policy().model)
        # # run until episode ends
        episode_reward = 0
        done = False

        if args.record:
            record_video(agent)
        else:
            eval_env = eval_env_creator(None)
            # create a video writer with imageio
            obs = eval_env.reset()
            for i in range(5 * 500):
                action = agent.compute_action(obs)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                eval_env.render()
                if done:
                    obs = eval_env.reset()

    else:
        stop = {
            "training_iteration": 1000
        }

        if checkpoint:
            analysis = tune.run("PPO",config=config, stop=stop, verbose=1, checkpoint_freq=5, name=args.name, restore=checkpoint)

        else:
            analysis = tune.run("PPO", config=config, stop=stop, verbose=1, checkpoint_freq=5, name=args.name)

        checkpoints = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial("episode_reward_mean"),
            metric="episode_reward_mean")
        print(checkpoints)
        ray.shutdown()
