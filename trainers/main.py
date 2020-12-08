import mysawyer_env, mybaxter_env
import gym, os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, \
    StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes
import argparse
import torch as th

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trainer')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-n', '--n_workers', type=int, default=0)
    parser.add_argument('-c', '--config', type=int, default=0)
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-o', '--offset', type=int, default=0)
    parser.add_argument('-s', '--sawyer', action='store_true')
    parser.add_argument('-p', '--pnp', action='store_true')
    args = parser.parse_args()
    algo = ''

    xml_file = 'reach' if not args.pnp else 'pnp'
    if args.sawyer:

        urdf_path = os.path.join(f'sawyer_description/urdf/{xml_file}.xml')

        starting_joint_angles = {'right_j0': -0.041662954890248294,
                                 'right_j1': -0.5158291091425074,
                                 'right_j2': 0.0203680414401436,
                                 'right_j3': 1.4,
                                 'right_j4': 0,
                                 'right_j5': 0.5,
                                 'right_j6': 1.7659649178699421}
        id = 'SawyerGym-v0'
    else:
        urdf_path = os.path.join(f'robots/baxter/{xml_file}.xml')

        starting_joint_angles = {'left_w0': 1.75,
                                 'left_w1': 1.50,
                                 'left_w2': -0.4999997247485215,
                                 'left_e0': -1.5,
                                 'left_e1': 1.5,
                                 'left_s0': -0.3,
                                 'left_s1': 0}
        id = 'BaxterGym-v0'
    Trainer = DDPG
    num_cpu = args.n_workers  # Number of processes to use
    hid_dim = [400, 300] if not args.pnp else [100, 100]
    act_fn = th.nn.Tanh
    batch_sz = 128
    ac = False
    policy_kwargs = dict
    # Create the vectorized environment

    eval_env = gym.make(id=id, model_path=urdf_path,
                        n_substeps=4,
                        gripper_extra_height=0.2,
                        block_gripper=args.pnp,
                        has_object=args.pnp,
                        target_in_the_air=True,
                        target_offset=0.0,
                        obj_range=0.2,
                        target_range=0.2,
                        distance_threshold=0.05,
                        initial_qpos=starting_joint_angles,
                        reward_type='dense')

    if not args.n_workers:
        print("Gym environment Running in Single Process Mode")
        if args.config == 0:
            Trainer = DDPG
            algo = 'DDPG'
        elif args.config == 1:
            Trainer = SAC
            algo = 'SAC'
        elif args.config == 2:
            Trainer = TD3
            algo = 'TD3'

        policy_kwargs = dict(net_arch=dict(pi=hid_dim, qf=hid_dim), activation_fn=act_fn)
        env = eval_env
    else:
        num_cpu = args.n_workers  # Number of processes to use
        print(f"Gym environment Running with {num_cpu} workers")
        # Create the vectorized environment
        env = make_vec_env(env_id=id, n_envs=num_cpu, env_kwargs=dict(model_path=urdf_path,
                                                                      n_substeps=4,
                                                                      gripper_extra_height=0.2,
                                                                      block_gripper=args.pnp,
                                                                      has_object=args.pnp,
                                                                      target_in_the_air=True,
                                                                      target_offset=0.0,
                                                                      obj_range=0.2,
                                                                      target_range=0.2,
                                                                      distance_threshold=0.05,
                                                                      initial_qpos=starting_joint_angles,
                                                                      reward_type='dense'),
                           vec_env_cls=SubprocVecEnv)
        hid_dim = [400, 300]
        Trainer = PPO
        ac = True
        algo = 'PPO'
        act_fn = th.nn.ReLU if args.config > 0 else th.nn.Tanh
        policy_kwargs = dict(net_arch=[dict(pi=hid_dim, vf=hid_dim)], activation_fn=act_fn)

    # Create log dir
    save_name = f"{id}_{algo}_{args.config}{'_pnp' if args.pnp else ''}"
    log_dir = "log/" + save_name
    os.makedirs(log_dir, exist_ok=True)
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5000, verbose=1)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-2 if not args.pnp else 350, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + '/best_model',
                                 log_path=log_dir + '/results', n_eval_episodes=10, eval_freq=1000, callback_on_new_best=callback_on_best,
                                 verbose=1, deterministic=not ac)
    callback = CallbackList([callback_max_episodes, eval_callback])
    info = f'hidden dim: {hid_dim}, batch_size: {batch_sz}, trainer: {Trainer}, actv_fn: {act_fn}'
    print('Running', save_name, ':', info)

    if args.load or args.eval:
        print('Loaded Model Successfully')
        model = Trainer.load(log_dir + '/best_model/best_model.zip', env)
    else:
        print('starting training')
        model = Trainer('MlpPolicy', env, batch_size=batch_sz, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=log_dir + '/tensor_board')
    if not args.eval:
        print('Running. Press CTRL-C to exit.')
        model.learn(total_timesteps=10000000, callback=callback, tb_log_name="first_run", reset_num_timesteps=True)
        model.save(save_name)

    del model

    print('Evaluating Model')
    model = Trainer.load(log_dir + '/best_model/best_model.zip', env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"Eval mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Enjoy trained agent
    obs = eval_env.reset()
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=not ac)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()
        if i % 300 == 0:
            obs = eval_env.reset()
