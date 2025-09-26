"""Neural networks used by SAC for the preset imitation task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.actions import ActionModule


class TanhNormal:
    """Distribution for tanh-transformed diagonal normal."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 2.0)
        self.std = self.log_std.exp()
        self.normal = torch.distributions.Normal(self.mean, self.std)

    def sample(self, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            pre_tanh = self.mean
        else:
            pre_tanh = self.normal.rsample()
        action = torch.tanh(pre_tanh)
        # Log probability following appendix C of SAC paper
        log_prob = self.normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


def mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    dims = [input_dim] + [hidden_dim] * (num_layers - 1)
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class SacActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        param_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.param_dim = param_dim
        self.num_modules = len(ActionModule)

        self.backbone = mlp(observation_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.module_head = nn.Linear(hidden_dim, self.num_modules)
        self.delta_mean = nn.Linear(hidden_dim, param_dim)
        self.delta_log_std = nn.Linear(hidden_dim, param_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        module_logits = self.module_head(h)
        mean = self.delta_mean(h)
        log_std = self.delta_log_std(h)
        return module_logits, mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        gumbel_tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        module_logits, mean, log_std = self.forward(obs)
        # Module selection via straight-through gumbel softmax for gradients
        module_probs = F.softmax(module_logits, dim=-1)
        gumbel_sample = F.gumbel_softmax(module_logits, tau=gumbel_tau, hard=True)
        module_indices = gumbel_sample.argmax(dim=-1)

        module_dist = torch.distributions.Categorical(probs=module_probs)
        module_log_prob = module_dist.log_prob(module_indices).unsqueeze(-1)

        tanh_normal = TanhNormal(mean, log_std)
        delta, delta_log_prob = tanh_normal.sample(deterministic=deterministic)

        log_prob = module_log_prob + delta_log_prob
        return module_indices, delta, log_prob, module_logits


class SacQNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        param_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_modules = len(ActionModule)
        input_dim = observation_dim + param_dim + self.num_modules
        self.backbone = mlp(input_dim, hidden_dim, 1, num_layers=num_layers)

    def forward(self, obs: torch.Tensor, delta: torch.Tensor, module_indices: torch.Tensor) -> torch.Tensor:
        module_one_hot = F.one_hot(module_indices, num_classes=self.num_modules).float()
        x = torch.cat([obs, delta, module_one_hot], dim=-1)
        q_value = self.backbone(x)
        return q_value
