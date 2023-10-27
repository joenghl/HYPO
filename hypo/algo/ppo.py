import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from .base import Algorithm
from hypo.utils import Normalization, RewardScaling, disable_gradient
from hypo.buffer import RolloutBuffer
from hypo.network import StateIndependentPolicy, StateFunction, PPOPolicy


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


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, logger, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_a=3e-4, lr_c=3e-4, units_actor=(64, 64),
                 units_critic=(64, 64), epoch_ppo=10, clip_eps=0.2, lam=0.97, coef_ent=0.00, mini_batch=64,
                 max_grad_norm=10.0, max_steps=1e6, use_lr_decay=False, use_reward_scaler=False,
                 use_state_norm=False, **kwargs):
        super().__init__(state_shape, action_shape, device, seed, logger, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
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

        self.reward_scaler = RewardScaling(shape=1, gamma=gamma)
        self.state_normalizer = Normalization(shape=state_shape)

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.lr_a_now = lr_a
        self.lr_c_now = lr_c
        self.optim_actor = Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        self.logger = logger
        self.max_steps = max_steps
        self.use_lr_decay = use_lr_decay
        self.use_reward_scaler = use_reward_scaler
        self.use_state_norm = use_state_norm

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.mini_batch = mini_batch
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lam = lam
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        if self.use_state_norm:
            next_state = self.state_normalizer(next_state)
        if self.use_reward_scaler:
            reward = self.reward_scaler(reward)
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done or t >= env._max_episode_steps:
            t = 0
            next_state = env.reset()
            if self.use_state_norm:
                next_state = self.state_normalizer(next_state)
            if self.use_reward_scaler:
                self.reward_scaler.reset()

        return next_state, t

    def update(self, step):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, step)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   step):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lam)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            for index in BatchSampler(SubsetRandomSampler(range(self.rollout_length)), self.mini_batch, False):
                self.update_critic(states[index], targets[index], step)
                self.update_actor(states[index], actions[index], log_pis[index], gaes[index], step)

        if self.use_lr_decay:
            self.lr_decay(step)

    def update_critic(self, states, targets, step):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, gaes, step):
        dist = self.actor.get_dist(states)
        log_pis = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        loss_actor_total = loss_actor - self.coef_ent * entropy

        self.optim_actor.zero_grad()
        loss_actor_total.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

    def lr_decay(self, step):
        self.lr_a_now = self.lr_a * (1 - step / self.max_steps)
        self.lr_c_now = self.lr_c * (1 - step / self.max_steps)
        for p in self.optim_actor.param_groups:
            p['lr'] = self.lr_a_now
        for p in self.optim_critic.param_groups:
            p['lr'] = self.lr_c_now

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


class PPOExpert(PPO):
    def __init__(self, state_shape, action_shape, device, path, seed, logger, units_actor=(64, 64), **kwargs):
        super().__init__(state_shape, action_shape, device, seed, logger, units_actor, **kwargs)
        self.actor = PPOPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor.load_state_dict(torch.load(path))

        disable_gradient(self.actor)
        self.device = device
