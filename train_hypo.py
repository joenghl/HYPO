import os
import argparse
from datetime import datetime
import torch
from tensorboardX import SummaryWriter

from hypo.env import make_env
from hypo.buffer import SerializedBuffer
from hypo.algo import HYPO
from hypo.trainer import Trainer


def run():
    env = make_env(args.env_train)
    env_test = make_env(args.env_eval)
    logger = SummaryWriter()
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    )

    algo = HYPO(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(f"cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
        logger=logger,
        log_interval=args.log_interval,
        rollout_length=args.rollout_length,
        lr_a=args.lr_a,
        lr_c=args.lr_c,
        lr_disc=args.lr_disc,
        step_kl=args.step_kl,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        step_max=args.num_steps,
        epoch_disc=args.epoch_disc,
        epoch_ppo=args.epoch_ppo,
        use_lr_decay=args.use_lr_decay,
        coef_kl=args.coef_kl,
        coef_ent=args.coef_ent,
        epoch_bc=args.epoch_bc,
        step_eta=args.step_eta,
        kl_min=args.kl_min
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_train, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        logger=logger,
        log_interval=args.log_interval,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed
    )
    trainer.train()
    logger.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=4096)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--log_interval', type=int, default=10000)
    p.add_argument('--eval_interval', type=int, default=10000)
    p.add_argument('--save_interval', type=int, default=10000)
    p.add_argument('--env_train', type=str, default='HalfCheetahSparseEnv')
    p.add_argument('--env_eval', type=str, default='HalfCheetah-v2')
    p.add_argument('--algo', type=str, default='hypo')
    p.add_argument('--lr_a', type=float, default=3e-4)
    p.add_argument('--lr_c', type=float, default=3e-4)
    p.add_argument('--lr_disc', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--mini_batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use_lr_decay', action='store_true')
    p.add_argument('--coef_ent', type=float, default=0.01)
    p.add_argument('--epoch_disc', type=int, default=10)
    p.add_argument('--epoch_ppo', type=int, default=10)
    p.add_argument('--step_kl', type=int, default=5e5)
    p.add_argument('--coef_kl', type=float, default=1.0)
    p.add_argument('--epoch_bc', type=int, default=20)
    p.add_argument('--step_eta', type=int, default=5e5)
    p.add_argument('--kl_min', type=float, default=0.02)

    args = p.parse_args()
    run()
