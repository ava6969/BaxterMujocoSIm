from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics import rotations
import utils
import numpy as np
import math
import os
from gym.spaces.box import Box
from stable_baselines3.common.env_checker import check_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BaxterEnv(RobotEnv):
    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
    ):
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
        self.observation_space = Box(-np.inf, np.inf, (23,), np.float64)
        self.action_space = Box(-1, 1, (9,), np.float64)

        self.constraints = {"left_s0": 50,
                            "left_s1": 50,
                            "left_e0": 50,
                            "left_e1": 50,
                            "left_w0": 15,
                            "left_w1": 15,
                            "left_w2": 15,
                            "l_grip": [0, 0.020833],
                            "r_grip": [-0.020833, 0]
                            }

        super(BaxterEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=7,
            initial_qpos=initial_qpos)

    def compute_reward(self, achieved_goal:np, goal:np, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        tot = 0
        for t_a in self.torque_applied:
            tot += t_a*t_a

        excess_torque_penalty = math.sqrt(tot) * self.c_a

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d - excess_torque_penalty

    def _step_callback(self):
        pass

    def _get_obs(self):
        # representing gripper for now - l_gripper_r_finger
        grip_pos = self.sim.data.get_body_xpos('l_gripper_r_finger')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('l_gripper_r_finger') * dt
        ll, rr = self.sim.data.get_joint_qpos('l_gripper_l_finger_joint'), \
                 self.sim.data.get_joint_qpos('l_gripper_r_finger_joint')

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

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        np_obs = np.concatenate([robot_qpos.ravel(), robot_qvel.ravel(), [ll], [rr], self.goal])
        return {'observation': np_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy}

    def _set_action(self, action):
        assert action.shape == (9,)  # 7 - joints 2 - l r gripper position
        action = action.copy()
        torques, l_r_gripper_pos = action[:7], action[7:]
        self.torque_applied = torques
        actions = {}
        for (k, v), a in zip(self.constraints.items()[:-2], torques):
            actions[k] = a * v

        l_min, l_max = self.constraints['l_grip']
        r_min, r_max = self.constraints['r_grip']
        l_grip_pos, r_grip_pos = self.unnormalize(l_r_gripper_pos[0], l_min, l_max), \
                                 self.unnormalize(l_r_gripper_pos[1], r_min, r_max)

        utils.ctrl_set_action(self.sim, actions.values())
        self.sim.data.set_joint_qpos('l_gripper_l_finger_joint', l_grip_pos)
        self.sim.data.set_joint_qpos('l_gripper_r_finger_joint', r_grip_pos)

    def estimate_ee(self):
        l_grip_pos = self.sim.data.get_joint_qpos('l_gripper_l_finger_joint' )
        r_grip_pos = self.sim.data.get_joint_qpos('l_gripper_r_finger_joint')
        return r_grip_pos - l_grip_pos

    def reset(self):
        obs = super(BaxterEnv, self).reset()
        print(len(obs['observation']))
        print(self.observation_space)
        return obs['observation']

    def step(self, action):
        obs, reward, done, info = super(BaxterEnv, self).step(action)
        done = info['is_success']
        print(len(obs['observation']))
        return obs['observation'], reward, done, info

    def unnormalize(self, x, min_x, max_x):
        return (x + 1)(max_x - min_x) / 2

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

    def _env_setup(self, initial_qpos):
        for (k, v) in initial_qpos.items():
            self.sim.data.set_joint_qpos(k, v)
        self.sim.forward()

        # Move end effector into position.
        # todo

        # Extract information for sampling goals.
        # grip = self.estimate_ee()
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('l_gripper_r_finger').copy()
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


if __name__ == '__main__':
    urdf_path = os.path.join('/assets/baxter_description/urdf/pick_n_place.xml')
    starting_joint_angles = {'left_w0': 1.75,
                             'left_w1': 1.50,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.5,
                             'left_e1': 1.5,
                             'left_s0': -0.3,
                             'left_s1': 0}
    initial_goal = [0, 0, 0]
    env = BaxterEnv(model_path=urdf_path,
                    n_substeps=20,
                    gripper_extra_height=0.2,
                    block_gripper=True,
                    has_object=False,
                    target_in_the_air=True,
                    target_offset=0.0,
                    obj_range=0.2,
                    target_range=0.2,
                    distance_threshold=0.05,
                    initial_qpos=starting_joint_angles,
                    reward_type='dense')
    check_env(env)
