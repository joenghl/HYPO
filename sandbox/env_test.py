from envs.mujoco_sparse import HopperSparseEnvV1, HopperSparseEnvV2, SwimmerSparseEnv, HalfCheetahSparseEnv, \
                               Walker2dSparseEnv, AntSparseEnv
from gail_airl_ppo.env import NormalizedEnv
import numpy as np
import gym


def run_test(env, num_loop=100):
    total_reward = 0.
    for i in range(num_loop):
        # print(f'Running loop {i}:')
        s = env.reset()
        # print(f"Init state: {s}.")
        done = False
        t = 0

        while not done and t <= env._max_episode_steps:
            t += 1
            action = env.action_space.sample()
            s, reward, done, info = env.step(action)
            total_reward += reward
            # print(f"step: {t},    action: {action},    state: {s},    reward: {reward},    done: {done}.")
    print(f'Mean Reward:  {total_reward/num_loop}')


if __name__ == "__main__":
    # test_env = AntSparseEnv(sparse_threshold=2.)
    test_env = 'Ant-v2'
    # test_env = NormalizedEnv(test_env)
    test_env = NormalizedEnv(gym.make(test_env))
    run_test(test_env)
