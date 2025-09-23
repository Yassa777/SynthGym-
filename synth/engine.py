"""TorchSynth-powered Serum-lite style synthesizer wrapper used by the RL environment.

This module exposes a compact API so the RL code can stay agnostic to the
underlying synthesis engine.

Key entry points
----------------
`SerumLiteSynth.get_param_vector()`
    Returns the current preset as a normalized `[0, 1]` tensor.
`SerumLiteSynth.set_param_vector(x)`
    Updates the preset using normalized parameters (clamped to `[0, 1]`).
`SerumLiteSynth.render(midi_note, dur_s)`
    Renders an audio buffer (batch size 1) for a given MIDI note and duration.

The implementation relies on TorchSynth building blocks (ADSR, LFO, Noise,
ControlRateUpsample, VCA) and custom lightweight signal processing for
waveform morphing, unison detune, and a state-variable filter.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torchsynth.config import SynthConfig
from torchsynth.module import ADSR, ControlRateUpsample, LFO, MonophonicKeyboard, Noise, VCA
from torchsynth.signal import Signal

# Prefer explicit type hints for clarity.
AudioTensor = Tensor  # Shape: (batch=1, n_samples)


@dataclass(frozen=True)
class ParamSpec:
    """Metadata describing a single normalized synthesizer parameter."""

    name: str
    minimum: float
    maximum: float
    default: float
    scale: str = "linear"  # Supported: "linear", "log"
    dtype: str = "float"  # Supported: "float", "int"
    description: str = ""

    def denormalize(self, value: Tensor) -> Tensor:
        """Convert `[0, 1]` value(s) into the parameter's physical range."""
        if value.ndim == 0:
            value = value.unsqueeze(0)

        value = value.clamp(0.0, 1.0)
        if self.scale == "linear":
            scaled = self.minimum + (self.maximum - self.minimum) * value
        elif self.scale == "log":
            if self.minimum <= 0.0 or self.maximum <= 0.0:
                raise ValueError(f"Log scale requires positive bounds for {self.name}.")
            log_min = math.log(self.minimum)
            log_max = math.log(self.maximum)
            scaled = torch.exp(log_min + (log_max - log_min) * value)
        else:
            raise ValueError(f"Unsupported scale '{self.scale}' for {self.name}.")

        if self.dtype == "int":
            scaled = torch.round(scaled).clamp(self.minimum, self.maximum)
        elif self.dtype != "float":
            raise ValueError(f"Unsupported dtype '{self.dtype}' for {self.name}.")
        return scaled


class SerumLiteSynth:
    """Compact Serum-lite style synthesizer built on TorchSynth primitives."""

    #: Continuous parameters exposed to the RL environment (normalized to [0, 1]).
    PARAM_SPECS: Tuple[ParamSpec, ...] = (
        ParamSpec("osc1_wave_morph", 0.0, 1.0, 0.0, description="OSC1 waveform morphing: sine → saw → square → triangle."),
        ParamSpec("osc1_unison", 1, 7, 0.0, dtype="int", description="OSC1 unison voices (1-7)."),
        ParamSpec("osc1_detune_cents", 0.0, 50.0, 0.1, description="OSC1 unison detune spread in cents."),
        ParamSpec("osc1_level", 0.0, 1.0, 0.7, description="OSC1 output gain."),
        ParamSpec("osc2_wave_morph", 0.0, 1.0, 0.5, description="OSC2 waveform morphing: sine → saw → square → triangle."),
        ParamSpec("osc2_unison", 1, 7, 0.0, dtype="int", description="OSC2 unison voices (1-7)."),
        ParamSpec("osc2_detune_cents", 0.0, 50.0, 0.1, description="OSC2 unison detune spread in cents."),
        ParamSpec("osc2_level", 0.0, 1.0, 0.6, description="OSC2 output gain."),
        ParamSpec("osc2_semitone_offset", -12.0, 12.0, 0.5, description="OSC2 coarse tuning in semitones."),
        ParamSpec("noise_level", 0.0, 1.0, 0.1, description="White noise mix level."),
        ParamSpec("filter_cutoff", 30.0, 20000.0, 0.5, scale="log", description="State-variable filter cutoff in Hz."),
        ParamSpec("filter_resonance", 0.1, 10.0, 0.2, description="State-variable filter resonance (Q)."),
        ParamSpec("filter_env_amount", -1.0, 1.0, 0.6, description="Depth of modulation envelope mapped to cutoff (± range)."),
        ParamSpec("amp_attack", 0.001, 2.0, 0.1, scale="log", description="Amplitude envelope attack (s)."),
        ParamSpec("amp_decay", 0.005, 2.0, 0.2, scale="log", description="Amplitude envelope decay (s)."),
        ParamSpec("amp_sustain", 0.0, 1.0, 0.8, description="Amplitude envelope sustain level."),
        ParamSpec("amp_release", 0.02, 4.0, 0.3, scale="log", description="Amplitude envelope release (s)."),
        ParamSpec("mod_attack", 0.001, 2.0, 0.05, scale="log", description="Modulation envelope attack (s)."),
        ParamSpec("mod_decay", 0.005, 2.0, 0.2, scale="log", description="Modulation envelope decay (s)."),
        ParamSpec("mod_sustain", 0.0, 1.0, 0.5, description="Modulation envelope sustain level."),
        ParamSpec("mod_release", 0.02, 4.0, 0.3, scale="log", description="Modulation envelope release (s)."),
        ParamSpec("lfo_rate", 0.05, 15.0, 0.2, scale="log", description="LFO rate in Hz."),
        ParamSpec("lfo_amount", 0.0, 1.0, 0.2, description="LFO modulation depth."),
        ParamSpec("lfo_target", 0.0, 1.0, 0.0, description="Crossfade between pitch (0) and cutoff (1) modulation."),
        ParamSpec("master_volume", 0.0, 1.0, 0.7, description="Overall output gain."),
    )

    #: Maximum per-sample cutoff modulation in octaves contributed by envelopes.
    _ENV_CUTOFF_RANGE_OCT = 4.0
    #: Maximum pitch modulation depth (± semitones) when LFO fully targets pitch.
    _LFO_PITCH_RANGE_SEMITONES = 7.0
    #: Maximum cutoff modulation depth in octaves when LFO fully targets filter.
    _LFO_CUTOFF_RANGE_OCT = 2.5

    def __init__(
        self,
        sample_rate: int = 16_000,
        buffer_seconds: float = 0.8,
        control_rate: int = 200,
        device: Optional[torch.device] = None,
    ) -> None:
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive.")

        self.device = device or torch.device("cpu")
        self.synthconfig = SynthConfig(
            batch_size=1,
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_seconds,
            control_rate=control_rate,
            reproducible=False,
        )
        self.synthconfig.to(self.device)

        # TorchSynth building blocks reused directly.
        self.keyboard = MonophonicKeyboard(self.synthconfig, device=self.device)
        self.amp_env = ADSR(self.synthconfig, device=self.device)
        self.mod_env = ADSR(self.synthconfig, device=self.device)
        self.lfo = LFO(self.synthconfig, device=self.device)
        self.control_upsample = ControlRateUpsample(self.synthconfig, device=self.device)
        self.vca = VCA(self.synthconfig, device=self.device)
        self.noise = Noise(self.synthconfig, device=self.device, seed=37)

        # Cache normalized parameters (always stored in [0, 1]).
        self._params = torch.tensor(
            [spec.default for spec in self.PARAM_SPECS], dtype=torch.float32, device=self.device
        )

        # Preallocate time vectors to avoid recomputing every render.
        self._audio_time = (
            torch.arange(self.synthconfig.buffer_size.item(), device=self.device, dtype=torch.float32)
            / self.synthconfig.sample_rate
        ).unsqueeze(0)  # Shape: (1, buffer)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_param_vector(self) -> Tensor:
        """Return the current preset as a normalized `[num_params]` tensor."""
        return self._params.clone().detach()

    def set_param_vector(self, values: Sequence[float | Tensor]) -> None:
        """Update all parameters from normalized values, clamping to `[0, 1]`."""
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.shape != self._params.shape:
            raise ValueError(
                f"Expected {self._params.numel()} parameters, received {tensor.numel()}."
            )
        self._params = tensor.clamp(0.0, 1.0)

    def to_preset_dict(self) -> Dict[str, float]:
        """Serialize the current preset to a JSON-friendly dictionary."""
        return {
            spec.name: float(value)
            for spec, value in zip(self.PARAM_SPECS, self._params.tolist())
        }

    def get_denormalized_params(self) -> Dict[str, float]:
        """Expose the current preset in physical units useful for debugging/visuals."""
        return self._denormalized_params()

    def load_preset_dict(self, preset: Mapping[str, float]) -> None:
        """Load parameters from a preset dictionary (normalized values)."""
        ordered = []
        for spec in self.PARAM_SPECS:
            if spec.name not in preset:
                raise KeyError(f"Preset is missing parameter '{spec.name}'.")
            ordered.append(float(preset[spec.name]))
        self.set_param_vector(ordered)

    def save_preset(self, path: Path | str) -> None:
        """Persist the current preset as JSON."""
        Path(path).write_text(json.dumps(self.to_preset_dict(), indent=2))

    def load_preset(self, path: Path | str) -> None:
        """Load preset values from a JSON file."""
        self.load_preset_dict(json.loads(Path(path).read_text()))

    @property
    def sample_rate(self) -> int:
        return int(self.synthconfig.sample_rate.item())

    @property
    def buffer_size(self) -> int:
        return int(self.synthconfig.buffer_size.item())

    @property
    def buffer_seconds(self) -> float:
        return float(self.synthconfig.buffer_size_seconds.item())

    def render(self, midi_note: float, dur_s: float) -> AudioTensor:
        """Render a single-note audio clip using the current preset."""
        if dur_s <= 0:
            raise ValueError("dur_s must be positive.")

        dur_s = min(dur_s, self.buffer_seconds)
        midi_note = float(midi_note)

        midi_tensor = torch.tensor([midi_note], dtype=torch.float32, device=self.device)
        dur_tensor = torch.tensor([dur_s], dtype=torch.float32, device=self.device)
        self.keyboard.set_parameter("midi_f0", midi_tensor)
        self.keyboard.set_parameter("duration", dur_tensor)
        midi_f0, note_on_duration = self.keyboard()

        params = self._denormalized_params()
        amp_env = self._configure_env(self.amp_env, params, prefix="amp", note_on_duration=note_on_duration)
        mod_env = self._configure_env(self.mod_env, params, prefix="mod", note_on_duration=note_on_duration)

        lfo_signal = self._configure_lfo(params)
        lfo_audio = self.control_upsample(lfo_signal)

        cutoff_signal = self._build_cutoff_envelope(params, mod_env, lfo_audio)
        pitch_mod_semitones = self._build_pitch_modulation(params, lfo_audio)

        freq_base = self._midi_to_hz(midi_f0)
        osc1 = self._render_oscillator(
            freq_base=freq_base,
            pitch_mod_semitones=pitch_mod_semitones,
            params=params,
            osc_prefix="osc1",
        )
        osc2 = self._render_oscillator(
            freq_base=freq_base * self._semitones_to_ratio(params["osc2_semitone_offset"]),
            pitch_mod_semitones=pitch_mod_semitones,
            params=params,
            osc_prefix="osc2",
        )
        noise = self.noise().clone()
        noise.uniform_(-1.0, 1.0)

        # Mix oscillators with envelopes.
        amp_env_audio = self.control_upsample(amp_env)
        osc_mix = (
            params["osc1_level"] * osc1
            + params["osc2_level"] * osc2
            + params["noise_level"] * noise
        )
        filtered = self._state_variable_filter(osc_mix, cutoff_signal, params["filter_resonance"])
        shaped = self.vca(filtered, amp_env_audio)

        return (params["master_volume"] * shaped).squeeze(0).detach().cpu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _denormalized_params(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for idx, spec in enumerate(self.PARAM_SPECS):
            denorm = spec.denormalize(self._params[idx])
            values[spec.name] = float(denorm.item())
        return values

    def _configure_env(
        self,
        env: ADSR,
        params: Mapping[str, float],
        prefix: str,
        note_on_duration: Tensor,
    ) -> Tensor:
        env.set_parameter("attack", torch.tensor([params[f"{prefix}_attack"]], device=self.device))
        env.set_parameter("decay", torch.tensor([params[f"{prefix}_decay"]], device=self.device))
        env.set_parameter("sustain", torch.tensor([params[f"{prefix}_sustain"]], device=self.device))
        env.set_parameter("release", torch.tensor([params[f"{prefix}_release"]], device=self.device))
        env.set_parameter("alpha", torch.tensor([2.0], device=self.device))
        return env(note_on_duration)

    def _configure_lfo(self, params: Mapping[str, float]) -> Tensor:
        self.lfo.set_parameter("frequency", torch.tensor([params["lfo_rate"]], device=self.device))
        self.lfo.set_parameter("mod_depth", torch.tensor([0.0], device=self.device))
        self.lfo.set_parameter("initial_phase", torch.tensor([0.0], device=self.device))
        # Focus the selector on a pure sine LFO.
        for shape in ("sin", "tri", "saw", "rsaw", "sqr"):
            value = 1.0 if shape == "sin" else 0.0
            self.lfo.set_parameter(shape, torch.tensor([value], device=self.device))
        return self.lfo()

    def _build_pitch_modulation(self, params: Mapping[str, float], lfo_audio: Tensor) -> Tensor:
        lfo_to_pitch = 1.0 - params["lfo_target"]
        depth = params["lfo_amount"] * lfo_to_pitch * self._LFO_PITCH_RANGE_SEMITONES
        return depth * lfo_audio

    def _build_cutoff_envelope(
        self,
        params: Mapping[str, float],
        mod_env: Tensor,
        lfo_audio: Tensor,
    ) -> Tensor:
        base_cutoff = params["filter_cutoff"]
        env_centered = (self.control_upsample(mod_env) - 0.5) * 2.0
        env_ratio = 2.0 ** (env_centered * params["filter_env_amount"] * self._ENV_CUTOFF_RANGE_OCT)
        lfo_to_cutoff = params["lfo_target"]
        lfo_ratio = 2.0 ** (lfo_audio * params["lfo_amount"] * lfo_to_cutoff * self._LFO_CUTOFF_RANGE_OCT)
        nyquist = 0.49 * self.synthconfig.sample_rate
        cutoff = torch.clamp(
            base_cutoff * env_ratio * lfo_ratio,
            min=30.0,
            max=float(nyquist),
        )
        return cutoff

    def _render_oscillator(
        self,
        freq_base: Tensor,
        pitch_mod_semitones: Tensor,
        params: Mapping[str, float],
        osc_prefix: str,
    ) -> Tensor:
        wave_param = params[f"{osc_prefix}_wave_morph"]
        unison = int(params[f"{osc_prefix}_unison"])
        detune_cents = params[f"{osc_prefix}_detune_cents"]
        level = params[f"{osc_prefix}_level"]

        voices = max(unison, 1)
        detune_offsets = torch.linspace(
            -1.0, 1.0, voices, device=self.device, dtype=torch.float32
        )
        detune_offsets = detune_offsets * detune_cents / 1200.0
        voice_ratios = 2.0 ** detune_offsets

        freq_per_voice = freq_base.unsqueeze(1) * voice_ratios.unsqueeze(0)
        pitch_ratio = 2.0 ** (pitch_mod_semitones / 12.0)
        freq_time = freq_per_voice.unsqueeze(-1) * pitch_ratio

        phase_increments = (2.0 * math.pi * freq_time) / self.synthconfig.sample_rate
        phase = torch.cumsum(phase_increments, dim=-1)
        phase = torch.remainder(phase, 2.0 * math.pi)

        sine_wave = torch.sin(phase)
        sine = sine_wave
        saw = 2.0 * ((phase / math.pi) % 2.0) - 1.0
        square = torch.sign(sine_wave)
        eps = 1e-6
        triangle = 2.0 / math.pi * torch.arcsin(torch.clamp(sine_wave, -1.0 + eps, 1.0 - eps))

        morph = wave_param
        morph_tensor = torch.tensor([morph], device=self.device, dtype=torch.float32)
        blend = morph_tensor * 3.0
        weights = torch.stack(
            [
                torch.clamp(1.0 - blend, min=0.0, max=1.0),
                torch.clamp(1.0 - (blend - 1.0).abs(), min=0.0, max=1.0),
                torch.clamp(1.0 - (blend - 2.0).abs(), min=0.0, max=1.0),
                torch.clamp(blend - 2.0, min=0.0, max=1.0),
            ],
            dim=-1,
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights.view(1, 1, 1, 4)

        waves = torch.stack([sine, saw, square, triangle], dim=-1)
        morphed = torch.sum(waves * weights, dim=-1)

        mix = morphed.mean(dim=1)  # Average unison voices.
        return (level * mix).as_subclass(Signal)

    def _state_variable_filter(
        self,
        audio: Tensor,
        cutoff_hz: Tensor,
        resonance: float,
    ) -> Tensor:
        # Transposed direct form II SVF per Zavalishin for stability.
        g = torch.tan(math.pi * cutoff_hz / self.synthconfig.sample_rate)
        g = torch.clamp(g, max=50.0)
        k = 1.0 / max(resonance, 1e-5)
        h = 1.0 / (1.0 + g * (g + k))

        batch, n_samples = audio.shape
        hp = torch.zeros(batch, device=audio.device)
        bp = torch.zeros(batch, device=audio.device)
        lp = torch.zeros(batch, device=audio.device)
        out = torch.zeros_like(audio)

        for idx in range(n_samples):
            g_step = g[:, idx]
            h_step = h[:, idx]
            x = audio[:, idx]
            hp = (x - (k + g_step) * bp - lp) * h_step
            bp = g_step * hp + bp
            lp = g_step * bp + lp
            out[:, idx] = lp

        return out.as_subclass(Signal)

    @staticmethod
    def _midi_to_hz(midi: Tensor) -> Tensor:
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    @staticmethod
    def _semitones_to_ratio(semitones: float) -> float:
        return 2.0 ** (semitones / 12.0)


__all__ = ["SerumLiteSynth", "ParamSpec"]
