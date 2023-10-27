import os
import argparse
import torch

from hypo.env import make_env
from hypo.algo import EXP
from hypo.utils import collect_demo


def run(args):
    env = make_env(args.env_id)

    algo = EXP[args.expert_algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(f"cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"),
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device(f"cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
        state_norm=args.state_norm
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}_rwd{args.reward}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--expert_algo', type=str, default='tppo')
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='Hopper-v2')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--state_norm', action='store_true')
    p.add_argument('--reward', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
