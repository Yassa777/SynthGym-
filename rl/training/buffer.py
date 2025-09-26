"""Experience replay buffer used by the SAC trainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class TransitionBatch:
    observations: torch.Tensor
    actions: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Simple FIFO buffer storing environment interactions."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        param_dim: int,
        device: torch.device,
        *,
        num_modules: int,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.observation_dim = observation_dim
        self.param_dim = param_dim
        self.num_modules = num_modules

        self.ptr = 0
        self.full = False

        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.modules = np.zeros((capacity,), dtype=np.int64)
        self.deltas = np.zeros((capacity, param_dim), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        module: int,
        delta: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.observations[idx] = observation
        self.next_observations[idx] = next_observation
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.modules[idx] = module
        self.deltas[idx] = delta

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size() < batch_size:
            raise ValueError("Not enough samples in buffer to satisfy batch size")
        indices = np.random.randint(0, self.size(), size=batch_size)

        obs = torch.from_numpy(self.observations[indices]).to(self.device)
        next_obs = torch.from_numpy(self.next_observations[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).unsqueeze(-1).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).unsqueeze(-1).to(self.device)
        modules = torch.from_numpy(self.modules[indices]).to(self.device)
        deltas = torch.from_numpy(self.deltas[indices]).to(self.device)

        action_dict = {
            "module": modules,
            "delta": deltas,
        }
        return TransitionBatch(
            observations=obs,
            actions=action_dict,
            rewards=rewards,
            next_observations=next_obs,
            dones=dones,
        )

    def to(self, device: torch.device) -> "ReplayBuffer":
        self.device = device
        return self
