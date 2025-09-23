"""Observation assembly utilities for the preset imitation environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .features import AudioFeatures


@dataclass(frozen=True)
class ObservationSlices:
    """Pointer structure describing observation vector components."""

    logmel_current: slice
    logmel_target: slice
    mfcc_delta: slice
    spectral_centroid_delta: slice
    loudness_delta: slice
    params: slice
    total_dim: int

    def as_dict(self) -> Dict[str, slice]:
        return {
            "logmel_current": self.logmel_current,
            "logmel_target": self.logmel_target,
            "mfcc_delta": self.mfcc_delta,
            "spectral_centroid_delta": self.spectral_centroid_delta,
            "loudness_delta": self.loudness_delta,
            "params": self.params,
        }


class ObservationBuilder:
    """Produce flattened observations combining audio features and synth parameters."""

    def __init__(self) -> None:
        self._slices: ObservationSlices | None = None

    @property
    def slices(self) -> ObservationSlices:
        if self._slices is None:
            raise RuntimeError("ObservationBuilder.build must be called before accessing slices.")
        return self._slices

    def build(
        self,
        current: AudioFeatures,
        target: AudioFeatures,
        params: torch.Tensor,
    ) -> torch.Tensor:
        if params.ndim != 2:
            raise ValueError("params tensor must have shape (batch, param_dim)")
        batch = params.shape[0]

        logmel_current = current.log_mel.flatten(1)
        logmel_target = target.log_mel.flatten(1)

        mfcc_delta = torch.mean(current.mfcc - target.mfcc, dim=2)
        centroid_delta = (
            torch.mean(current.spectral_centroid - target.spectral_centroid, dim=1, keepdim=True)
        )
        loudness_delta = (current.loudness - target.loudness).unsqueeze(1)

        components = (
            ("logmel_current", logmel_current),
            ("logmel_target", logmel_target),
            ("mfcc_delta", mfcc_delta),
            ("spectral_centroid_delta", centroid_delta),
            ("loudness_delta", loudness_delta),
            ("params", params),
        )

        for name, tensor in components:
            if tensor.shape[0] != batch:
                raise ValueError(f"Component {name} batch size mismatch: {tensor.shape[0]} != {batch}")

        observation = torch.cat([tensor for _, tensor in components], dim=1)

        # Record slices for downstream usage
        offset = 0
        slice_map: Dict[str, slice] = {}
        for name, tensor in components:
            width = tensor.shape[1]
            slice_map[name] = slice(offset, offset + width)
            offset += width

        self._slices = ObservationSlices(
            logmel_current=slice_map["logmel_current"],
            logmel_target=slice_map["logmel_target"],
            mfcc_delta=slice_map["mfcc_delta"],
            spectral_centroid_delta=slice_map["spectral_centroid_delta"],
            loudness_delta=slice_map["loudness_delta"],
            params=slice_map["params"],
            total_dim=observation.shape[1],
        )

        return observation


__all__ = ["ObservationBuilder", "ObservationSlices"]
