import torch
import wandb
import torch.nn as nn

from gail_airl_ppo.env import make_env
from gail_airl_ppo.network import StateDependentPolicy, PPOPolicy


def run_loop(env, agent, num_loop=20):
    total_reward = 0.
    for i in range(1, num_loop+1):
        s = env.reset()
        done = False
        episode_reward = 0.
        while not done:
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float)
                action = agent(s.unsqueeze_(0)).numpy()
            s, reward, done, info = env.step(action)
            episode_reward += reward
        print(f'Loop {i} reward: {episode_reward}')
        wandb.log({'return/eval_expert': episode_reward}, step=i)
        total_reward += episode_reward

    mean_reward = total_reward / num_loop
    print(f'Mean reward: {mean_reward:.2f}')
    wandb.log({'return/mean_reward': mean_reward})


if __name__ == "__main__":
    wandb.init(project='eval_expert', name='Ant-ppo-expert-test')
    PATH = '/workspace/win/gail/logs/Ant-v2/tppo/seed0-20230424-1331/model/step1048000/actor.pth'
    eval_env = make_env('Ant-v2')
    algo = 'ppo'
    if algo == 'ppo':
        actor = PPOPolicy(
            state_shape=eval_env.observation_space.shape,
            action_shape=eval_env.action_space.shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to('cpu')
    elif algo == 'sac':
        actor = StateDependentPolicy(
            state_shape=eval_env.observation_space.shape,
            action_shape=eval_env.action_space.shape,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU()
        ).to('cpu')
    actor.load_state_dict(torch.load(PATH))
    run_loop(eval_env, actor)
