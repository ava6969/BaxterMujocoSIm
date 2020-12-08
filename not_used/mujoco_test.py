import mujoco_py
import os
import time
import utils
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import numpy as np
mj_path, _ = mujoco_py.utils.discover_mujoco()
urdf_path = os.path.join('/assets/sawyer_description/urdf/reach.xml')
model = mujoco_py.load_model_from_path(urdf_path)
sim = mujoco_py.MjSim(model)

viewer = MjViewer(sim)

# commands

starting_joint_angles = {'right_j0': -0.041662954890248294,
                         'right_j1': -0.5158291091425074,
                         'right_j2': 0.0203680414401436,
                         'right_j3': 1.4,
                         'right_j4': 0,
                         'right_j5': 0.5,
                         'right_j6': 1.7659649178699421}

for (k, v) in starting_joint_angles.items():
    sim.data.set_joint_qpos(k, v)

x, v = utils.robot_get_obs(sim)
print(x)
print(v)
names = [n for n in sim.model.joint_names if n.startswith('l')]
print(names)
#print(sim.data.get_body_xpos('r_gripper_l_finger'))
# actuator
print(sim.data.qacc)
print(sim.data.qpos)

print(sim.data.qacc)
print(sim.data.qpos)

print(sim.data.ctrl)
print(sim.model.actuator_biastype)
#
while True:
    action = np.random.uniform(-1, 1, size=9)
    # for i in range(9):
    #     sim.data.ctrl[i] = action[i]
    print(sim.data.ctrl)
    sim.step()
    viewer.render()

# can control torques
# can get vel and pos
#