import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntSparseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, sparse_threshold=2.):
        self.sparse_threshold = sparse_threshold
        print(f"Using Sparse Ant Env, the sparse_threshold is: {self.sparse_threshold}.")
        self._max_episode_steps = 1000
        self.reward_flags = np.ones(10000, dtype=bool)
        self.max_level = 0
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        # old reward
        # forward_reward = (xposafter - xposbefore)/self.dt
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # new reward
        level = int((xposafter - self.init_qpos[0]) / self.sparse_threshold)
        if level >= 1 and self.reward_flags[level]:
            reward = 1.
            self.reward_flags[level] = False
        else:
            reward = 0.
        if level > self.max_level:
            self.max_level = level

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        # new added
        self.reward_flags = np.ones(10000, dtype=bool)
        self.max_level = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
