"""
Hybrid Policy Optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from .hypo_bc import HBC
from .hypo_ppo import HPPO
from .base import Algorithm
from hypo.network.disc import HYPODiscrim


class HYPO(Algorithm):
    def __init__(self, buffer_exp, state_shape, action_shape, device, seed, logger, gamma=0.995,
                 log_interval=1e3, lr_bc=1e-3, lr_disc=3e-4, lr_a=3e-4, lr_c=3e-4, batch_size=256,
                 units_bc=(64, 64), units_disc=(64, 64), rollout_length=2048, start_step=4e3, mini_batch=256,
                 epoch_bc=20, epoch_disc=10, step_kl=2e6, step_max=1e7, coef_kl=1.0, eta_min=0.2, eta_max=0.8,
                 step_eta=2e6, coef_ent=0.00, kl_min=0.1, use_lr_decay=False, **kwargs):
        super().__init__(state_shape, action_shape, device, seed, logger, gamma)
        self.batch_bc = batch_size
        self.start_step = start_step
        self.logger = logger
        self.log_interval = log_interval
        self.device = device
        self.epoch_bc = epoch_bc
        self.epoch_disc = epoch_disc
        self.step_kl = step_kl
        self.eta = eta_min
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.step_eta = step_eta

        self.disc = HYPODiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        # policies
        self.policy_bc = HBC(buffer_exp, state_shape, action_shape, device, seed, logger,
                             log_interval=log_interval, lr_actor=lr_bc, batch_size=batch_size,
                             units_actor=units_bc)
        self.policy_ppo = HPPO(state_shape, action_shape, device, seed, logger, rollout_length=rollout_length,
                               lr_a=lr_a, lr_c=lr_c, step_kl=step_kl, step_max=step_max, coef_kl=coef_kl,
                               use_lr_decay=use_lr_decay, mini_batch=mini_batch, coef_ent=coef_ent, kl_min=kl_min)

        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)

        # Make the initial weights of bc_actor and ppo_actor consistent.
        self.policy_ppo.actor.load_state_dict(self.policy_bc.actor.state_dict())

    def exploit(self, state):
        return self.policy_ppo.exploit(state)

    def step(self, env,  state, t, step):
        dist = self.policy_bc.actor.get_dist(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device))
        return self.policy_ppo.step(env, state, t, dist.loc.detach(), dist.scale.detach(), step)

    def is_update(self, step):
        return step % self.policy_ppo.rollout_length == 0

    def update(self, step):
        for _ in range(self.epoch_disc):
            # update disc
            with torch.no_grad():
                # get exp (s,a,log_pi)
                states_exp, actions_exp = self.policy_bc.buffer_exp.sample(self.batch_bc)[:2]
                log_pis_exp = self.policy_bc.actor.get_dist(states_exp).log_prob(actions_exp)
                info_exp = (states_exp, actions_exp, log_pis_exp)
                # get agent (s,a,log_pi)
                states_ppo, actions_ppo = self.policy_ppo.buffer.sample(self.batch_bc)[:2]
                log_pis_ppo = self.policy_bc.actor.get_dist(states_ppo).log_prob(actions_ppo)
                info_ppo = (states_ppo, actions_ppo, log_pis_ppo)
            self.update_disc(info_exp, info_ppo, step)
        for _ in range(self.epoch_bc):
            self.update_bc(self.policy_ppo.buffer, step)
        self.update_ppo(step)
        self.update_eta(step)

    def update_bc(self, buffer_ppo, step):
        return self.policy_bc.update(buffer_ppo, self.disc.d, self.eta, step)

    def update_ppo(self, step):
        return self.policy_ppo.update(step)

    def update_disc(self, info_exp, info_ppo, step):
        # logits output between (-inf,inf), ppo->-inf, exp->inf.
        logits_exp = self.disc(info_exp[0], info_exp[1], info_exp[2])
        logits_ppo = self.disc(info_ppo[0], info_ppo[1], info_ppo[2])

        loss_exp = self.eta * -F.logsigmoid(logits_exp).mean()
        loss_ppo = -F.logsigmoid(-logits_ppo).mean()
        loss_extra = self.eta * F.logsigmoid(-logits_exp).mean()

        loss_disc = loss_exp + loss_ppo + loss_extra

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

    def update_eta(self, step):
        self.eta = np.clip(self.eta_max * (step / self.step_eta), self.eta_min, self.eta_max)

    def save_models(self, save_dir):
        pass
