import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperSparseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Sparse reward hopper env.
    Agent get +1. reward when reaching the new level.
    """
    def __init__(self, sparse_threshold=2.):
        self.sparse_threshold = sparse_threshold
        print(f"Using Sparse Hopper Env, the sparse_threshold is: {self.sparse_threshold}.")
        self._max_episode_steps = 1000
        self.reward_flags = np.ones(10000, dtype=bool)
        self.max_level = 0
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        # old reward
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()

        # new sparse reward
        level = int((posafter - self.init_qpos[0]) / self.sparse_threshold)
        if level >= 1 and self.reward_flags[level]:
            reward = 1.
            self.reward_flags[level] = False
        else:
            reward = 0.

        if level > self.max_level:
            self.max_level = level

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        # new added
        self.reward_flags = np.ones(10000, dtype=bool)
        self.max_level = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
