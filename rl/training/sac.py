"""Soft Actor-Critic agent and training loop for preset imitation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.actions import ActionModule
from rl.config import EnvConfig, FeatureExtractorConfig, PerceptualLossConfig
from rl.env import PresetImitationEnv
from rl.training.buffer import ReplayBuffer
from rl.training.networks import SacActor, SacQNetwork


@dataclass
class SacConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 0.1
    hidden_dim: int = 512
    num_layers: int = 3
    batch_size: int = 128
    updates_per_step: int = 1
    warmup_steps: int = 2000
    target_entropy_scale: float = 0.5
    gumbel_tau: float = 1.0


class SacAgent:
    def __init__(
        self,
        observation_dim: int,
        param_dim: int,
        config: SacConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.actor = SacActor(observation_dim, param_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(device)
        self.q1 = SacQNetwork(observation_dim, param_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(device)
        self.q2 = SacQNetwork(observation_dim, param_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(device)
        self.target_q1 = SacQNetwork(observation_dim, param_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(device)
        self.target_q2 = SacQNetwork(observation_dim, param_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(np.log(config.init_temperature), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -config.target_entropy_scale * (param_dim + len(ActionModule))

        self.param_dim = param_dim
        self._step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> dict:
        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            module_idx, delta, _, _ = self.actor.sample(obs_tensor, deterministic=deterministic, gumbel_tau=self.config.gumbel_tau)
            delta = delta.clamp(-1.0, 1.0)
        return {
            "module": module_idx.cpu().numpy().astype(np.int64),
            "delta": delta.cpu().numpy().astype(np.float32),
        }

    def update(self, buffer: ReplayBuffer) -> dict:
        stats = {}
        if buffer.size() < self.config.batch_size:
            return stats

        batch = buffer.sample(self.config.batch_size)
        obs = batch.observations
        next_obs = batch.next_observations
        rewards = batch.rewards
        dones = batch.dones
        module = batch.actions["module"].long()
        delta = batch.actions["delta"].float()

        with torch.no_grad():
            next_module, next_delta, next_log_prob, _ = self.actor.sample(next_obs, deterministic=False, gumbel_tau=self.config.gumbel_tau)
            next_delta = next_delta.clamp(-1.0, 1.0)
            q1_next = self.target_q1(next_obs, next_delta, next_module)
            q2_next = self.target_q2(next_obs, next_delta, next_module)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.config.gamma * min_q_next

        q1_pred = self.q1(obs, delta, module)
        q2_pred = self.q2(obs, delta, module)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=10.0)
        self.q1_opt.step()
        self.q2_opt.step()

        # Actor update
        new_module, new_delta, log_prob, _ = self.actor.sample(obs, deterministic=False, gumbel_tau=self.config.gumbel_tau)
        new_delta = new_delta.clamp(-1.0, 1.0)
        q1_new = self.q1(obs, new_delta, new_module)
        q2_new = self.q2(obs, new_delta, new_module)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        stats.update(
            critic_loss=float(critic_loss.detach().cpu().item()),
            actor_loss=float(actor_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
        )
        return stats

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.config.tau)
            tgt_param.data.add_(self.config.tau * src_param.data)


class SacTrainer:
    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        agent: SacAgent,
        buffer: ReplayBuffer,
        *,
        device: torch.device,
    ) -> None:
        self.env = gym.vector.SyncVectorEnv(env_fns)
        self.agent = agent
        self.buffer = buffer
        self.device = device
        self.num_envs = len(env_fns)
        self.obs, self.info = self.env.reset(seed=None)

    def train(
        self,
        total_steps: int,
        *,
        log_every: int = 100,
        eval_fn: Callable[[SacAgent], dict] | None = None,
    ) -> List[dict]:
        logs: List[dict] = []
        for step in range(total_steps):
            if step < self.agent.config.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(self.obs, deterministic=False)
            next_obs, rewards, terminated, truncated, info = self.env.step(action)
            dones = np.logical_or(terminated, truncated)
            final_obs = info.get("final_observation", None)
            for env_idx in range(self.num_envs):
                next_observation = next_obs[env_idx]
                if final_obs is not None and dones[env_idx]:
                    candidate = final_obs[env_idx]
                    if candidate is not None:
                        next_observation = candidate
                self.buffer.add(
                    self.obs[env_idx],
                    int(action["module"][env_idx]),
                    action["delta"][env_idx],
                    float(rewards[env_idx]),
                    next_observation,
                    bool(dones[env_idx]),
                )
            self.obs = next_obs

            stats = {}
            for _ in range(self.agent.config.updates_per_step):
                latest = self.agent.update(self.buffer)
                if latest:
                    stats = latest
            if step % log_every == 0:
                entry = {"step": step}
                entry.update(stats)
                if eval_fn is not None:
                    entry.update({f"eval/{k}": v for k, v in eval_fn(self.agent).items()})
                logs.append(entry)
        return logs


def make_env_fns(
    num_envs: int,
    env_config: EnvConfig,
    feature_config: FeatureExtractorConfig | None = None,
    loss_config: PerceptualLossConfig | None = None,
) -> List[Callable[[], gym.Env]]:
    fns: List[Callable[[], gym.Env]] = []
    for _ in range(num_envs):
        def _make_env(env_config=env_config, feature_config=feature_config, loss_config=loss_config):
            return PresetImitationEnv(
                env_config=env_config,
                feature_config=feature_config,
                loss_config=loss_config,
            )

        fns.append(_make_env)
    return fns
