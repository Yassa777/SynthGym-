"""Perceptual loss computation and tracking utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import PerceptualLossConfig, PerceptualLossWeights
from .features import AudioFeatures, FeatureExtractor


@dataclass
class LossBreakdown:
    """Structured result of a perceptual loss evaluation."""

    total: torch.Tensor
    terms: Dict[str, torch.Tensor]
    bonus: torch.Tensor
    reward: torch.Tensor
    improved: torch.Tensor
    features: AudioFeatures


class _PerceptualLoss:
    """Internal helper that computes weighted perceptual losses."""

    def __init__(self, weights: PerceptualLossWeights) -> None:
        self.weights = weights

    def __call__(self, current: AudioFeatures, target: AudioFeatures) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        components: Dict[str, torch.Tensor] = {}

        mr_losses = []
        for cur_mag, tgt_mag in zip(current.mrstft_magnitude, target.mrstft_magnitude):
            diff = torch.abs(cur_mag - tgt_mag)
            mr_losses.append(diff.mean(dim=(1, 2)))
        mrstft_loss = torch.stack(mr_losses, dim=0).mean(dim=0)
        components["mrstft"] = mrstft_loss

        mfcc_loss = torch.mean(torch.abs(current.mfcc - target.mfcc), dim=(1, 2))
        components["mfcc"] = mfcc_loss

        centroid_current = torch.mean(current.spectral_centroid, dim=1)
        centroid_target = torch.mean(target.spectral_centroid, dim=1)
        centroid_loss = torch.abs(centroid_current - centroid_target)
        components["spectral_centroid"] = centroid_loss

        loudness_loss = torch.abs(current.loudness - target.loudness)
        components["loudness"] = loudness_loss

        clip_penalty = current.clip_ratio
        components["clip_penalty"] = clip_penalty

        dc_penalty = torch.abs(current.dc_offset)
        components["dc_penalty"] = dc_penalty

        total = (
            self.weights.mrstft * mrstft_loss
            + self.weights.mfcc * mfcc_loss
            + self.weights.spectral_centroid * centroid_loss
            + self.weights.loudness * loudness_loss
            + self.weights.clip_penalty * clip_penalty
            + self.weights.dc_penalty * dc_penalty
        )

        return total, components


class PerceptualEvaluator:
    """Stateful evaluator that tracks best loss and improvement bonuses."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        config: PerceptualLossConfig | None = None,
    ) -> None:
        self.extractor = feature_extractor
        self.config = config or PerceptualLossConfig()
        self.loss_fn = _PerceptualLoss(self.config.weights)

        self.target_features: AudioFeatures | None = None
        self.target_loudness: torch.Tensor | None = None
        self.best_loss: torch.Tensor | None = None
        self.best_audio: torch.Tensor | None = None

    def reset(self, target_audio: torch.Tensor) -> AudioFeatures:
        """Analyze the target audio and reset the tracking state."""
        with torch.no_grad():
            target_features = self.extractor.analyze(target_audio)
        self.target_features = target_features
        self.target_loudness = target_features.loudness.detach()
        batch = target_audio.shape[0]
        device = target_features.processed_audio.device
        self.best_loss = torch.full((batch,), float("inf"), device=device)
        self.best_audio = target_features.processed_audio.detach().clone()
        return target_features

    def evaluate(self, current_audio: torch.Tensor) -> LossBreakdown:
        if self.target_features is None or self.target_loudness is None or self.best_loss is None:
            raise RuntimeError("PerceptualEvaluator.reset must be called before evaluate.")

        features = self.extractor.analyze(current_audio, match_loudness=self.target_loudness)
        total, terms = self.loss_fn(features, self.target_features)

        if self.config.clamp_min_loss is not None:
            total = total.clamp_min(self.config.clamp_min_loss)
        if self.config.clamp_max_loss is not None:
            total = total.clamp_max(self.config.clamp_max_loss)

        improved = total < self.best_loss

        bonus = torch.zeros_like(total)
        if self.config.improvement_bonus > 0.0:
            prev = self.best_loss
            finite_prev = torch.where(torch.isfinite(prev), prev, torch.zeros_like(prev))
            diff = torch.clamp(finite_prev - total, min=0.0)
            bonus = self.config.improvement_bonus * diff
            bonus = torch.where(improved, bonus, torch.zeros_like(bonus))

        self.best_loss = torch.where(improved, total.detach(), self.best_loss)
        processed_audio = features.processed_audio.detach()
        if self.best_audio is None:
            self.best_audio = processed_audio.clone()
        else:
            improved_mask = improved.unsqueeze(1)
            self.best_audio = torch.where(improved_mask, processed_audio, self.best_audio)

        reward = -(total - bonus)

        return LossBreakdown(
            total=total,
            terms=terms,
            bonus=bonus,
            reward=reward,
            improved=improved,
            features=features,
        )


__all__ = ["PerceptualEvaluator", "LossBreakdown"]
