"""Shared configuration dataclasses for the RL preset imitation stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


@dataclass(frozen=True)
class MRSTFTScale:
    """Configuration for a single multi-resolution STFT scale."""

    fft_size: int
    hop_size: int
    win_length: int
    window: str = "hann"


@dataclass(frozen=True)
class FeatureExtractorConfig:
    """Parameters controlling audio feature extraction."""

    sample_rate: int = 16_000
    logmel_n_mels: int = 64
    logmel_win_length: int = 1024
    logmel_hop_length: int = 256
    logmel_fmin: float = 20.0
    logmel_fmax: float | None = None
    mfcc_n_mfcc: int = 20
    spectral_n_fft: int = 1024
    spectral_hop_length: int = 256
    loudness_block_size: float = 0.4
    loudness_hop_size: float = 0.2
    apply_softclip: bool = True
    target_loudness: float | None = None  # LUFS; set at runtime if curriculum-specific.
    mrstft_scales: Tuple[MRSTFTScale, ...] = field(
        default_factory=lambda: (
            MRSTFTScale(fft_size=2048, hop_size=512, win_length=2048),
            MRSTFTScale(fft_size=1024, hop_size=256, win_length=1024),
            MRSTFTScale(fft_size=512, hop_size=128, win_length=512),
        )
    )


@dataclass(frozen=True)
class PerceptualLossWeights:
    """Scalar weights applied to each perceptual loss component."""

    mrstft: float = 1.0
    mfcc: float = 0.5
    spectral_centroid: float = 0.2
    loudness: float = 0.2
    clip_penalty: float = 0.1
    dc_penalty: float = 0.05


@dataclass(frozen=True)
class PerceptualLossConfig:
    """High-level configuration for perceptual loss evaluation."""

    weights: PerceptualLossWeights = PerceptualLossWeights()
    improvement_bonus: float = 0.0
    clamp_min_loss: float | None = None
    clamp_max_loss: float | None = None


@dataclass(frozen=True)
class EnvConfig:
    """High-level configuration shared by scalar and vector environments."""

    max_steps: int = 50
    success_threshold: float = 0.02
    delta_scale: float = 0.08
    curriculum_stage: int = 0
    midi_note: int = 60
    render_duration: float = 0.5


__all__ = [
    "MRSTFTScale",
    "FeatureExtractorConfig",
    "PerceptualLossWeights",
    "PerceptualLossConfig",
    "EnvConfig",
]
