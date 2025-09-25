"""Vectorized Gymnasium environment for Serum-lite preset imitation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from synth.engine import ParamSpec, SerumLiteSynth

from .actions import (
    ActionModule,
    CurriculumStage,
    DEFAULT_CURRICULUM,
    action_mask_for,
    param_dim,
    parameter_indices_for,
)
from .config import EnvConfig, FeatureExtractorConfig, PerceptualLossConfig
from .features import AudioFeatures, FeatureExtractor
from .observation import ObservationBuilder
from .perceptual import PerceptualEvaluator


class PresetImitationVecEnv(gym.vector.VectorEnv):
    """Vectorized environment operating on multiple presets in parallel."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        *,
        env_config: EnvConfig | None = None,
        feature_config: FeatureExtractorConfig | None = None,
        loss_config: PerceptualLossConfig | None = None,
        curriculum: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.env_config = env_config or EnvConfig()
        self.curriculum = curriculum
        if not curriculum:
            raise ValueError("Curriculum must contain at least one stage.")
        self.stage_index = min(self.env_config.curriculum_stage, len(curriculum) - 1)
        self.stage = self.curriculum[self.stage_index]

        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.param_dim = param_dim()
        self.param_specs: Sequence[ParamSpec] = SerumLiteSynth.PARAM_SPECS

        # Individual synth instances (still rendered sequentially, but state isolated).
        self.synths = [SerumLiteSynth(sample_rate=(feature_config.sample_rate if feature_config else 16_000)) for _ in range(num_envs)]
        self.feature_extractor = FeatureExtractor(feature_config)
        self.perceptual = PerceptualEvaluator(self.feature_extractor, loss_config)
        self.observation_builder = ObservationBuilder()

        default_params = torch.tensor([spec.default for spec in self.param_specs], dtype=torch.float32)
        self._param_tensor = default_params.unsqueeze(0).repeat(num_envs, 1)
        self._target_params = self._param_tensor.clone()

        self._active_mask = action_mask_for(self.stage)
        self._discrete_indices = torch.tensor(
            [idx for idx, spec in enumerate(self.param_specs) if spec.dtype == "int"],
            dtype=torch.long,
        )

        obs_sample = self._bootstrap_observation()
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32,
        )
        action_space = spaces.Dict(
            {
                "module": spaces.MultiDiscrete(np.full(self.num_envs, len(ActionModule), dtype=np.int64)),
                "delta": spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs, self.param_dim), dtype=np.float32),
            }
        )

        self.observation_space = observation_space
        self.action_space = action_space

        self._rng = np.random.default_rng()
        self._steps = np.zeros(self.num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # VectorEnv API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps[:] = 0

        self._target_params = torch.rand((self.num_envs, self.param_dim), dtype=torch.float32)
        target_audio = self._render_batch(self._target_params)
        self.perceptual.reset(target_audio)

        default_params = torch.tensor([spec.default for spec in self.param_specs], dtype=torch.float32)
        self._param_tensor = default_params.unsqueeze(0).repeat(self.num_envs, 1)
        current_audio = self._render_batch(self._param_tensor)
        result = self.perceptual.evaluate(current_audio)
        observation = self._build_observation(result.features)

        info = {
            "loss": result.total.detach().cpu().numpy(),
            "reward": result.reward.detach().cpu().numpy(),
            "best_loss": self.perceptual.best_loss.detach().cpu().numpy(),
            "stage": np.array([self.stage.name] * self.num_envs),
        }
        return observation, info

    def step(self, action: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        module_ids = np.asarray(action["module"], dtype=np.int64)
        deltas = torch.as_tensor(action["delta"], dtype=torch.float32)
        if deltas.shape != (self.num_envs, self.param_dim):
            raise ValueError(f"delta must have shape ({self.num_envs}, {self.param_dim})")

        for env_idx in range(self.num_envs):
            module = list(ActionModule)[module_ids[env_idx] % len(ActionModule)]
            self._apply_action(env_idx, module, deltas[env_idx])

        current_audio = self._render_batch(self._param_tensor)
        result = self.perceptual.evaluate(current_audio)

        observation = self._build_observation(result.features)
        loss_value = result.total.detach().cpu().numpy()
        reward = result.reward.detach().cpu().numpy()

        self._steps += 1
        terminated = loss_value <= self.env_config.success_threshold
        truncated = self._steps >= self.env_config.max_steps

        info = {
            "loss": loss_value,
            "reward": reward,
            "bonus": result.bonus.detach().cpu().numpy(),
            "improved": result.improved.detach().cpu().numpy().astype(bool),
            "best_loss": self.perceptual.best_loss.detach().cpu().numpy(),
            "stage": np.array([self.stage.name] * self.num_envs),
            "terms": {name: value.detach().cpu().numpy() for name, value in result.terms.items()},
        }

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bootstrap_observation(self) -> np.ndarray:
        dummy_audio = torch.zeros((self.num_envs, self.synths[0].buffer_size))
        self.perceptual.reset(dummy_audio)
        result = self.perceptual.evaluate(dummy_audio)
        obs = self._build_observation(result.features)
        return obs

    def _build_observation(self, current_features: AudioFeatures) -> np.ndarray:
        target_features = self.perceptual.target_features
        if target_features is None:
            raise RuntimeError("Target features not initialized.")
        obs_tensor = self.observation_builder.build(current_features, target_features, self._param_tensor)
        return obs_tensor.cpu().numpy().astype(np.float32)

    def _render_batch(self, params_batch: torch.Tensor) -> torch.Tensor:
        audio_buffers = []
        for idx, synth in enumerate(self.synths):
            params = params_batch[idx].clamp(0.0, 1.0)
            synth.set_param_vector(params.tolist())
            audio = synth.render(self.env_config.midi_note, self.env_config.render_duration)
            if isinstance(audio, torch.Tensor):
                tensor = audio
            else:
                tensor = torch.from_numpy(np.asarray(audio))
            audio_buffers.append(tensor)
        return torch.stack(audio_buffers, dim=0)

    def _apply_action(self, env_idx: int, module: ActionModule, delta: torch.Tensor) -> None:
        active_indices = torch.where(self._active_mask)[0]
        module_indices = parameter_indices_for(module)
        indices = torch.tensor(sorted(set(active_indices.tolist()) & set(module_indices.tolist())), dtype=torch.long)
        if indices.numel() == 0:
            return

        scaled_delta = delta[indices] * self.env_config.delta_scale
        self._param_tensor[env_idx, indices] = (
            self._param_tensor[env_idx, indices] + scaled_delta
        ).clamp(0.0, 1.0)
        self._snap_discrete(env_idx)

    def _snap_discrete(self, env_idx: int) -> None:
        if self._discrete_indices.numel() == 0:
            return
        for idx in self._discrete_indices.tolist():
            spec = self.param_specs[idx]
            value = self._param_tensor[env_idx, idx]
            denorm = spec.denormalize(value)
            if spec.maximum == spec.minimum:
                norm = torch.zeros_like(denorm)
            else:
                norm = (denorm - spec.minimum) / (spec.maximum - spec.minimum)
            self._param_tensor[env_idx, idx] = norm.clamp(0.0, 1.0)


__all__ = ["PresetImitationVecEnv"]
