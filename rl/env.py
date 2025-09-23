"""Gymnasium environment for Serum-lite preset imitation."""
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
from .config import FeatureExtractorConfig, PerceptualLossConfig
from .features import FeatureExtractor
from .observation import ObservationBuilder
from .perceptual import PerceptualEvaluator


@dataclass(frozen=True)
class EnvConfig:
    """High-level configuration for the preset imitation environment."""

    max_steps: int = 50
    success_threshold: float = 0.02
    delta_scale: float = 0.08
    curriculum_stage: int = 0
    midi_note: int = 60
    render_duration: float = 0.5


class PresetImitationEnv(gym.Env[np.ndarray, Dict[str, np.ndarray]]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_config: EnvConfig | None = None,
        feature_config: FeatureExtractorConfig | None = None,
        loss_config: PerceptualLossConfig | None = None,
        curriculum: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.env_config = env_config or EnvConfig()
        self.curriculum = curriculum
        if not curriculum:
            raise ValueError("Curriculum must contain at least one stage.")
        self.stage_index = min(self.env_config.curriculum_stage, len(curriculum) - 1)
        self.stage = self.curriculum[self.stage_index]

        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.param_dim = param_dim()
        self.param_specs: Sequence[ParamSpec] = SerumLiteSynth.PARAM_SPECS

        self.synth = SerumLiteSynth(sample_rate=feature_config.sample_rate if feature_config else 16_000)
        self.feature_extractor = FeatureExtractor(feature_config)
        self.perceptual = PerceptualEvaluator(self.feature_extractor, loss_config)
        self.observation_builder = ObservationBuilder()

        self._param_tensor = torch.tensor(
            [spec.default for spec in self.param_specs], dtype=torch.float32
        )
        self._target_params = self._param_tensor.clone()

        self._active_mask = action_mask_for(self.stage)
        self._discrete_indices = torch.tensor(
            [idx for idx, spec in enumerate(self.param_specs) if spec.dtype == "int"],
            dtype=torch.long,
        )

        # Bootstrap observation shape
        obs_sample = self._bootstrap_observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "module": spaces.Discrete(len(ActionModule)),
                "delta": spaces.Box(low=-1.0, high=1.0, shape=(self.param_dim,), dtype=np.float32),
            }
        )

        self._rng = np.random.default_rng()
        self._steps = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0

        self._target_params = torch.rand(self.param_dim)
        target_audio = self._render_from_params(self._target_params)
        self.perceptual.reset(target_audio.unsqueeze(0))

        self._param_tensor = torch.tensor(
            [spec.default for spec in self.param_specs], dtype=torch.float32
        )
        current_audio = self._render_from_params(self._param_tensor)
        result = self.perceptual.evaluate(current_audio.unsqueeze(0))
        observation = self._build_observation(result.features)

        info = {
            "loss": float(result.total.item()),
            "reward": float(result.reward.item()),
            "best_loss": float(self.perceptual.best_loss.item()),
            "stage": self.stage.name,
        }
        return observation, info

    def step(self, action: Dict[str, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, Dict]:
        module_id = int(action["module"]) % len(ActionModule)
        module = list(ActionModule)[module_id]
        delta = torch.as_tensor(action["delta"], dtype=torch.float32)
        if delta.shape != (self.param_dim,):
            raise ValueError(f"delta must have shape ({self.param_dim},)")

        self._apply_action(module, delta)

        current_audio = self._render_from_params(self._param_tensor)
        result = self.perceptual.evaluate(current_audio.unsqueeze(0))

        observation = self._build_observation(result.features)
        loss_value = float(result.total.item())
        reward = float(result.reward.item())

        self._steps += 1
        terminated = loss_value <= self.env_config.success_threshold
        truncated = self._steps >= self.env_config.max_steps

        info = {
            "loss": loss_value,
            "reward": reward,
            "bonus": float(result.bonus.item()),
            "improved": bool(result.improved.item()),
            "best_loss": float(self.perceptual.best_loss.item()),
            "stage": self.stage.name,
            "terms": {name: float(value.item()) for name, value in result.terms.items()},
        }

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bootstrap_observation(self) -> np.ndarray:
        dummy_audio = torch.zeros(1, self.synth.buffer_size)
        self.perceptual.reset(dummy_audio)
        result = self.perceptual.evaluate(dummy_audio)
        obs = self._build_observation(result.features)
        return obs

    def _build_observation(self, current_features) -> np.ndarray:
        target_features = self.perceptual.target_features
        if target_features is None:
            raise RuntimeError("Target features not initialized.")
        params = self._param_tensor.unsqueeze(0)
        obs_tensor = self.observation_builder.build(current_features, target_features, params)
        return obs_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def _render_from_params(self, params: torch.Tensor) -> torch.Tensor:
        params_clamped = params.clamp(0.0, 1.0)
        self.synth.set_param_vector(params_clamped.tolist())
        audio = self.synth.render(self.env_config.midi_note, self.env_config.render_duration)
        return torch.from_numpy(audio).to(params_clamped)

    def _apply_action(self, module: ActionModule, delta: torch.Tensor) -> None:
        active_indices = torch.where(self._active_mask)[0]
        module_indices = parameter_indices_for(module)
        indices = torch.tensor(sorted(set(active_indices.tolist()) & set(module_indices.tolist())), dtype=torch.long)
        if indices.numel() == 0:
            return

        scaled_delta = delta[indices] * self.env_config.delta_scale
        self._param_tensor[indices] = (self._param_tensor[indices] + scaled_delta).clamp(0.0, 1.0)
        self._snap_discrete()

    def _snap_discrete(self) -> None:
        if self._discrete_indices.numel() == 0:
            return
        for idx in self._discrete_indices.tolist():
            spec = self.param_specs[idx]
            value = self._param_tensor[idx]
            denorm = spec.denormalize(value)
            rounded = torch.round(denorm).clamp(spec.minimum, spec.maximum)
            if spec.maximum == spec.minimum:
                norm = torch.tensor(0.0)
            else:
                norm = (rounded - spec.minimum) / (spec.maximum - spec.minimum)
            self._param_tensor[idx] = norm.to(self._param_tensor.device)


__all__ = ["EnvConfig", "PresetImitationEnv"]
