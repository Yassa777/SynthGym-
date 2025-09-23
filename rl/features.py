"""Audio feature extraction utilities supporting the RL preset imitation environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torchaudio
from torchaudio import functional as F

from .config import FeatureExtractorConfig, MRSTFTScale


@dataclass
class AudioFeatures:
    """Container holding the feature tensors required by the environment."""

    processed_audio: torch.Tensor
    log_mel: torch.Tensor
    mfcc: torch.Tensor
    spectral_centroid: torch.Tensor
    loudness: torch.Tensor
    mrstft_magnitude: Tuple[torch.Tensor, ...]
    clip_ratio: torch.Tensor
    dc_offset: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "processed_audio": self.processed_audio,
            "log_mel": self.log_mel,
            "mfcc": self.mfcc,
            "spectral_centroid": self.spectral_centroid,
            "loudness": self.loudness,
            "mrstft_magnitude": torch.stack(self.mrstft_magnitude),
            "clip_ratio": self.clip_ratio,
            "dc_offset": self.dc_offset,
        }


class FeatureExtractor:
    """Compute perceptually-relevant audio features from waveform batches."""

    _EPS = 1e-6

    def __init__(
        self,
        config: FeatureExtractorConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or FeatureExtractorConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        self._build_transforms()
        self._build_mrstft_windows()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        audio: torch.Tensor,
        *,
        match_loudness: float | torch.Tensor | None = None,
    ) -> AudioFeatures:
        """Return the feature set for a batch of audio waveforms.

        Args:
            audio: Tensor shaped `(batch, num_samples)`.
            match_loudness: Optional LUFS target. If provided, the audio is
                loudness-normalized before feature computation.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected audio tensor with shape (batch, samples); got {tuple(audio.shape)}")

        audio = audio.to(device=self.device, dtype=self.dtype)
        processed = audio

        if match_loudness is not None:
            processed = self._match_loudness(processed, match_loudness)

        clip_ratio = (processed.abs() > 0.99).float().mean(dim=1)
        dc_offset = processed.mean(dim=1)

        if self.config.apply_softclip:
            processed = torch.tanh(processed)

        log_mel = self._log_mel(processed)
        mfcc = self._mfcc(processed)
        spectral_centroid = self._spectral_centroid(processed)
        loudness = self._loudness(processed)
        mrstft = self._multi_resolution_stft(processed)

        return AudioFeatures(
            processed_audio=processed,
            log_mel=log_mel,
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            loudness=loudness,
            mrstft_magnitude=mrstft,
            clip_ratio=clip_ratio,
            dc_offset=dc_offset,
        )

    def to(self, device: torch.device | str) -> "FeatureExtractor":
        """Move internal modules to the specified device."""
        self.device = torch.device(device)
        self.mel_spec = self.mel_spec.to(self.device)
        self.mfcc_transform = self.mfcc_transform.to(self.device)
        self.spectral_window = self.spectral_window.to(self.device)
        self._mrstft_windows = [window.to(self.device) for window in self._mrstft_windows]
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_transforms(self) -> None:
        cfg = self.config
        mel_kwargs = {
            "sample_rate": cfg.sample_rate,
            "n_fft": cfg.logmel_win_length,
            "win_length": cfg.logmel_win_length,
            "hop_length": cfg.logmel_hop_length,
            "f_min": cfg.logmel_fmin,
            "f_max": cfg.logmel_fmax,
            "n_mels": cfg.logmel_n_mels,
            "power": 2.0,
            "center": True,
        }

        self.mel_spec = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
        mfcc_melkwargs = {
            k: v for k, v in mel_kwargs.items() if k not in {"center", "sample_rate"}
        }
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=cfg.sample_rate,
            n_mfcc=cfg.mfcc_n_mfcc,
            melkwargs=mfcc_melkwargs,
        )
        self.spectral_window = torch.hann_window(
            cfg.spectral_n_fft, periodic=True, dtype=self.dtype
        ).to(self.device)
        self.mel_spec = self.mel_spec.to(self.device, dtype=self.dtype)
        self.mfcc_transform = self.mfcc_transform.to(self.device, dtype=self.dtype)

    def _build_mrstft_windows(self) -> None:
        self._mrstft_windows: List[torch.Tensor] = []
        for scale in self.config.mrstft_scales:
            if scale.window != "hann":
                raise ValueError(f"Unsupported window type {scale.window}")
            window = torch.hann_window(scale.win_length, periodic=True, dtype=self.dtype)
            self._mrstft_windows.append(window.to(self.device))

    def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(audio)
        return torch.log(mel + self._EPS)

    def _mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mfcc_transform(audio)

    def _spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        centroid = F.spectral_centroid(
            audio,
            cfg.sample_rate,
            pad=0,
            window=self.spectral_window,
            n_fft=cfg.spectral_n_fft,
            win_length=cfg.spectral_n_fft,
            hop_length=cfg.spectral_hop_length,
        )
        return torch.nan_to_num(centroid, nan=0.0)

    def _loudness(self, audio: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        rms = torch.sqrt(torch.mean(audio**2, dim=1) + self._EPS)
        lufs = 20.0 * torch.log10(rms + self._EPS)
        return lufs

    def _multi_resolution_stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        magnitudes: List[torch.Tensor] = []
        for scale, window in zip(self.config.mrstft_scales, self._mrstft_windows):
            spec = torch.stft(
                audio,
                n_fft=scale.fft_size,
                hop_length=scale.hop_size,
                win_length=scale.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            magnitudes.append(spec.abs())
        return tuple(magnitudes)

    def _match_loudness(
        self,
        audio: torch.Tensor,
        target_lufs: float | torch.Tensor,
        clamp_db: float = 24.0,
    ) -> torch.Tensor:
        batch = audio.shape[0]
        if isinstance(target_lufs, torch.Tensor):
            target = target_lufs.to(audio.device, dtype=self.dtype)
        else:
            target = torch.as_tensor(target_lufs, dtype=self.dtype, device=audio.device)
        if target.ndim == 0:
            target = target.expand(batch)
        elif target.shape[0] == 1 and batch > 1:
            target = target.expand(batch)

        current = self._loudness(audio)
        diff = target - current
        diff = diff.clamp(min=-clamp_db, max=clamp_db)
        gain = torch.pow(10.0, diff / 20.0)
        return audio * gain.unsqueeze(1)


__all__ = ["FeatureExtractor", "AudioFeatures"]
