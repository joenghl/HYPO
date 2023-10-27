from torch import nn
from torch.optim import Adam
import torch

from hypo.network import PPOPolicy
from .base import Algorithm


class HBC(Algorithm):
    def __init__(self, buffer_exp, state_shape, action_shape, device, seed, logger, gamma=0.995,
                 log_interval=1e3, lr_actor=3e-4, batch_size=64, units_actor=(64, 64), **kwargs):
        super().__init__(state_shape, action_shape, device, seed, logger, gamma)
        self.buffer_exp = buffer_exp
        self.batch_size = batch_size

        self.actor = PPOPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.log_interval = log_interval
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.logger = logger
        self.alpha = 6.0
        self.scale_exp = 10.0
        self.scale_ppo = 3.0

    @staticmethod
    def step(env, state, t, step):
        return None, t+1

    def is_update(self, step):
        return step >= self.batch_size

    def update(self, buffer_ppo, d, eta, step):
        self.learning_steps += 1
        states_exp, actions_exp = self.buffer_exp.sample(self.batch_size)[:2]
        states_ppo, actions_ppo = buffer_ppo.sample(self.batch_size)[:2]

        dist_exp = self.actor.get_dist(states_exp)
        dist_ppo = self.actor.get_dist(states_ppo)

        log_pis_exp = dist_exp.log_prob(actions_exp)
        log_pis_ppo = dist_ppo.log_prob(actions_ppo)

        entropy_exp = dist_exp.entropy().mean()
        entropy_ppo = dist_ppo.entropy().mean()

        # Weights calculated by discriminator.
        d_exp = torch.clamp(d(states_exp, actions_exp, log_pis_exp), 0.1, 0.9)
        d_ppo = torch.clamp(d(states_ppo, actions_ppo, log_pis_ppo), 0.1, 0.9)
        coef_exp = torch.clamp((self.alpha - eta * 1.0 / (d_exp.mean() * (1 - d_exp.mean()))) / self.scale_exp, 0)
        coef_ppo = torch.clamp((1.0 / (1.0 - d_ppo.mean())) / self.scale_ppo, 0)

        loss_bc_exp = -log_pis_exp.mean()
        loss_bc_ppo = -log_pis_ppo.mean()

        loss = loss_bc_exp * coef_exp + loss_bc_ppo * coef_ppo

        self.optim_actor.zero_grad()
        loss.backward()
        self.optim_actor.step()

    def save_models(self, save_dir):
        pass
