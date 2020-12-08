import gym
from gym.envs.robotics import rotations
from mujoco_py import MujocoException

import utils
import numpy as np
import math
import os
from gym.spaces.box import Box

import copy
import gym
from gym import error, spaces
from gym.utils import seeding

DEFAULT_SIZE = 500
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))
from gym.envs.registration import register

reg = register(
    id='SawyerGym-v0',
    entry_point='mysawyer_env:SawyerEnv',
    max_episode_steps=300,
)


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SawyerEnv(gym.Env):
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
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        self.observation_space = Box(-np.inf, np.inf, (27,), np.float32)
        self.action_space = Box(-1, 1, (7,), np.float32)

        self.constraints = [80, 80, 40, 40, 9, 9, 9, 20, 20]

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

    def compute_reward(self, achieved_goal: np, goal: np, info):
        # Compute distance between goal and the achieved goal.
        d = 0
        if not self.has_object:
            d = goal_distance(achieved_goal, goal)
        tot = 0
        for i, t_a in enumerate(self.torque_applied):
            t_a *= self.constraints[i]
            tot += t_a * t_a

        excess_torque_penalty = math.sqrt(tot) * self.c_a

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d - excess_torque_penalty

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('right_gripper_l_finger_joint', 0.)
            self.sim.data.set_joint_qpos('right_gripper_r_finger_joint', 0.)
            self.sim.forward()

    def _get_obs(self):
        # representing gripper for now - l_gripper_r_finger
        grip_pos = self.sim.data.get_body_xpos('right_gripper_l_finger')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('right_gripper_l_finger') * dt

        # instead of ee im getting both ll and rr xpos
        ll, rr = self.sim.data.get_body_xpos('right_gripper_l_finger'), \
                 self.sim.data.get_body_xpos('right_gripper_r_finger')

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim,'right')
        np_obs = np.concatenate([robot_qpos.ravel(), robot_qvel.ravel(), ll, rr, self.goal])
        return {'observation': np_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy}

    def _set_action(self, action):
        assert action.shape == (7,)
        action = action.copy()
        self.torque_applied = action
        utils.ctrl_set_action(self.sim, np.array(action))

    def estimate_ee(self):
        l_grip_pos = self.sim.data.get_joint_qpos('l_gripper_l_finger_joint')
        r_grip_pos = self.sim.data.get_joint_qpos('l_gripper_r_finger_joint')
        return r_grip_pos - l_grip_pos

    def unnormalize(self, x, min_x, max_x):
        return (x + 1) * (max_x - min_x) / 2

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + \
                   self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset

            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)

        return goal.copy()

    def _env_setup(self, initial_qpos:dict):

        pos_r = self.np_random.uniform(-0.02, 0.02, size=9)
        vel_r = self.np_random.uniform(-0.1, 0.1, size=9)
        for i, (k, v) in enumerate(initial_qpos.items()):
            self.sim.data.set_joint_qpos(k, v + pos_r[i])

        for i, k in enumerate(initial_qpos.keys()):
            self.sim.data.set_joint_qvel(k, vel_r[i])

        self.sim.forward()

        # Move end effector into position.
        # todo
        # Extract information for sampling goals.
        # grip = self.estimate_ee()
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('right_gripper_r_finger').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

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
        body_id = self.sim.model.body_name2id('right_gripper_l_finger')
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
        self.sim.model.site_pos[site_id] = temp
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

