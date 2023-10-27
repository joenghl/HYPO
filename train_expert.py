import os
import argparse
from datetime import datetime
import torch
from tensorboardX import SummaryWriter

from hypo.env import make_env
from hypo.algo import ALGOS
from hypo.trainer import Trainer


def run(args):
    env = make_env(args.env_train)
    env_test = make_env(args.env_eval)
    logger = SummaryWriter()
    algo = ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
        logger=logger,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        rollout_length=args.rollout_length,
        max_steps=args.num_steps,
        coef_ent=args.coef_ent
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_train, f'{args.algo}', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        logger=logger
    )
    trainer.train()
    logger.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_train', type=str, default='Hopper-v2')
    p.add_argument('--env_eval', type=str, default='Hopper-v2')
    p.add_argument('--num_steps', type=int, default=1e6)
    p.add_argument('--rollout_length', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mini_batch', type=int, default=64)
    p.add_argument('--log_interval', type=int, default=1e4)
    p.add_argument('--eval_interval', type=int, default=1e4)
    p.add_argument('--save_interval', type=int, default=1e4)
    p.add_argument('--coef_ent', type=float, default=0.001)
    p.add_argument('--algo', type=str, default='ppo', choices=['ppo'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    run(args)
