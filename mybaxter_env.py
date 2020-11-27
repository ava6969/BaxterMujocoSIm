import gym
from gym.envs.robotics import rotations
from mujoco_py import MujocoException

import utils
import numpy as np
import math
import os
from gym.spaces.box import Box
from stable_baselines3.common.env_checker import check_env

import copy
import gym
from gym import error, spaces
from gym.utils import seeding
DEFAULT_SIZE = 500
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from gym.envs.registration import register

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


reg = register(
    id='BaxterGym-v0',
    entry_point='mybaxter_env:BaxterEnv',
    max_episode_steps=300,
)


class BaxterEnv(gym.Env):
    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
    ):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.c_a = 1e-3  # regularization term
        self.torque_applied = []
        self.initial_gripper_xpos = None

        self.constraints = [50, 50, 50, 50, 15, 15, 15]
        self.gripper_open = [-0.0115, 0.0115]
        self.gripper_closed = [0.020833, -0.020833]
        self.l_gripper_range = [-0.020833, 0.0115]
        self.r_gripper_range = [-0.0115, 0.020833]
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()

        if self.has_object:
            self.observation_space = Box(-np.inf, np.inf, (25,), np.float32)
            self.action_space = Box(-1, 1, (8,), np.float32)
        else:
            self.observation_space = Box(-np.inf, np.inf, (20,), np.float32)
            self.action_space = Box(-1, 1, (7,), np.float32)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        done = info['is_success']
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs['observation'], reward, bool(done), info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs['observation']

    def compute_reward(self, achieved_goal:np, goal:np, info):
        # Compute distance between goal and the achieved goal.
        if self.has_object:
            obj_pos = self.sim.data.get_body_xpos('object0')
            ee_obj_dist = goal_distance(achieved_goal, obj_pos)
            target_obj_dist = goal_distance(obj_pos, goal)
            cube_half_width = 0.01
            lf_obj_dist = goal_distance(self.sim.data.get_body_xpos('l_finger').copy(), obj_pos)
            rf_obj_dist = goal_distance(self.sim.data.get_body_xpos('r_finger').copy(), obj_pos)
            fingers_dist = (lf_obj_dist - cube_half_width) + (rf_obj_dist - cube_half_width)
            d_w = ee_obj_dist + fingers_dist + target_obj_dist
            d = -1 / (1 + d_w)
        else:
            d = goal_distance(achieved_goal, goal)

        tot = 0
        for i, t_a in enumerate(self.torque_applied):
            tot += t_a * t_a

        excess_torque_penalty = math.sqrt(tot) * self.c_a

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d - excess_torque_penalty

    def _step_callback(self):
        pass

    def _get_obs(self):
        # representing gripper for now - l_gripper_r_finger
        grip_pos = self.sim.data.get_site_xpos('left_ee')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('left_ee') * dt

        if self.has_object:
            object_pos = self.sim.data.get_body_xpos('object0')
            # rotations
            # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            # object_velp = self.sim.data.get_site_xvelp('object0') * dt
            # object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            # object_rel_pos = object_pos - grip_pos
            # object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        if self.has_object:
            achieved_goal = grip_pos.copy()
            np_obs = np.concatenate([robot_qpos.ravel(), robot_qvel.ravel(), grip_pos, self.goal, object_pos])
        else:
            achieved_goal = grip_pos.copy()
            # achieved_goal = np.squeeze(object_pos.copy())
            np_obs = np.concatenate([robot_qpos.ravel(), robot_qvel.ravel(), grip_pos, self.goal])
        return {'observation': np_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy}

    def _set_action(self, action):
        if self.has_object:
            assert action.shape[0] == 8  # 7 - joints 2 - open or close
            if action[7] >= 1/3:
                self.sim.data.ctrl[7:] = self.gripper_closed
            elif action[7] <= -1/3:
                self.sim.data.ctrl[7:] = self.gripper_open

            # action[7] = self.unnormalize(action[7], self.l_gripper_range[0], self.l_gripper_range[1])
            # action[8] = self.unnormalize(action[8], self.l_gripper_range[0], self.l_gripper_range[1])
        else:
            assert action.shape[0] == 7  # 7 - joints

        action = action.copy()
        for i in range(7):
            action[i] = action[i]*self.constraints[i]

        self.torque_applied = action[:7]

        utils.ctrl_set_action(self.sim, np.array(action[:7])) # todo verify gripper movements

    def unnormalize(self, x, min_x, max_x):
        return (x + 1) * (max_x - min_x) / 2

    def _sample_goal(self):
        if self.has_object:
            goal = self.sim.data.get_site_xpos('target0')
            # goal = self.initial_gripper_xpos[:3] + \
            #        self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # goal += self.target_offset
            # goal[2] = self.height_offset
            #
            # if self.target_in_the_air and self.np_random.uniform() < 0.5:
            #     goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)

        return goal.copy()

    def _env_setup(self, initial_qpos):
        pos_r = self.np_random.uniform(-0.02, 0.02, size=7)
        vel_r = self.np_random.uniform(-0.1, 0.1, size=7)

        for i, (k, v) in enumerate(initial_qpos.items()):
            self.sim.data.set_joint_qpos(k, v + pos_r[i])
        for i, k in enumerate(initial_qpos.keys()):
            self.sim.data.set_joint_qvel(k, vel_r[i])

        self.sim.forward()

        # Move end effector into position.
        # todo
        # Extract information for sampling goals.
        # grip = self.estimate_ee()
        # for _ in range(10):
        #     self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('left_ee').copy()
        # if self.has_object:
        #     self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
        #                                                                              size=2)
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('left_hand')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        temp = self.goal - sites_offset[0]
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
