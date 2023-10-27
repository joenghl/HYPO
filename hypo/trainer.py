import os
from time import time, sleep
from datetime import timedelta


class Trainer:
    def __init__(self, env, env_test, algo, logger, log_interval, log_dir, seed=0, num_steps=1e5,
                 eval_interval=1e3, num_eval_episodes=5, save_interval=1e4):
        super().__init__()

        self.env = env
        self.env.seed(seed)

        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.logger = logger

        self.num_steps = int(num_steps)
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.num_eval_episodes = num_eval_episodes

        # Time to start training.
        self.start_time = time()

    def train(self):
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        if self.algo.use_state_norm:
            state = self.algo.state_normalizer(state)

        # Global timestep.
        for step in range(1, self.num_steps+1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(step)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
            if step % self.save_interval == 0:
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            if self.algo.use_state_norm:
                state = self.algo.state_normalizer(state)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                if self.algo.use_state_norm:
                    state = self.algo.state_normalizer(state)
                episode_return += reward
            mean_return += episode_return / self.num_eval_episodes
        self.logger.add_scalar('data/reward', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
