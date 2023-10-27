import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal, kl_divergence
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from .base import Algorithm
from hypo.utils import Normalization, RewardScaling
from hypo.buffer import RolloutBufferTypeA
from hypo.network import StateFunction, PPOPolicy


def calculate_gae(values, rewards, dones, next_values, gamma, lam):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lam * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class HPPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, logger, gamma=0.995, rollout_length=2048,
                 mini_batch=256, mix_buffer=20, lr_a=3e-4, lr_c=3e-4, units_actor=(64, 64), kl_min=0.1,
                 units_critic=(64, 64), epoch_ppo=10, clip_eps=0.2, lam=0.97, coef_ent=0.00, max_grad_norm=10.0,
                 coef_kl=1.0, step_kl=2e6, step_max=1e7, use_lr_decay=False):
        super().__init__(state_shape, action_shape, device, seed, logger, gamma)

        # Rollout buffer.
        self.buffer = RolloutBufferTypeA(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        self.actor = PPOPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.lr_a_now = lr_a
        self.lr_c_now = lr_c

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_a, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_c, eps=1e-5)
        self.state_normalizer = Normalization(shape=state_shape)
        self.reward_scaler = RewardScaling(shape=1, gamma=gamma)
        self.logger = logger

        self.use_lr_decay = use_lr_decay
        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.mini_batch = mini_batch
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lam = lam
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.coef_kl = coef_kl
        self.step_kl = step_kl
        self.step_max = step_max
        self.kl_min = kl_min

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, loc, scale, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi, next_state, loc, scale)

        if done or t >= env._max_episode_steps:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, step):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states, locs, scales = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, locs, scales, step)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, locs, scales, step):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lam)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            for index in BatchSampler(SubsetRandomSampler(range(self.rollout_length)), self.mini_batch, False):
                self.update_critic(states[index], targets[index], step)
                self.update_actor(states[index], actions[index], log_pis[index], gaes[index], locs[index],
                                  scales[index], step)
        if self.use_lr_decay:
            self.lr_decay(step)

    def update_critic(self, states, targets, step):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, gaes, locs, scales, step):
        dist = self.actor.get_dist(states)
        dist_bc = Normal(locs, scales)
        log_pis = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        loss_kl = kl_divergence(dist, dist_bc).mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        coef_kl_now = max(self.coef_kl * (1 - step / self.step_kl), self.kl_min)
        loss_actor_total = loss_actor - self.coef_ent * entropy + loss_kl * coef_kl_now

        self.optim_actor.zero_grad()
        loss_actor_total.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

    def lr_decay(self, step):
        self.lr_a_now = self.lr_a * (1 - step / self.step_max)
        self.lr_c_now = self.lr_c * (1 - step / self.step_max)
        for p in self.optim_actor.param_groups:
            p['lr'] = self.lr_a_now
        for p in self.optim_critic.param_groups:
            p['lr'] = self.lr_c_now

    def save_models(self, save_dir):
        pass
