import gym

from envs.mujoco_sparse import HopperSparseEnv, Walker2dSparseEnv, HalfCheetahSparseEnv, AntSparseEnv

gym.logger.set_level(40)


ENVS = {
    'HopperSparse': HopperSparseEnv,
    'Walker2dSparse': Walker2dSparseEnv,
    'HalfCheetahSparse': HalfCheetahSparseEnv,
    'AntSparse': AntSparseEnv
}


def make_env(env_id):
    if env_id not in ENVS.keys():
        return NormalizedEnv(gym.make(env_id))
    else:
        return NormalizedEnv(ENVS[env_id]())


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
