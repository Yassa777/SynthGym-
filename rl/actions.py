"""Action space helpers and curriculum-aware parameter masks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, List, Sequence

import torch

from synth.engine import SerumLiteSynth

_PARAM_INDEX = {spec.name: idx for idx, spec in enumerate(SerumLiteSynth.PARAM_SPECS)}
_PARAM_DIM = len(SerumLiteSynth.PARAM_SPECS)


class ActionModule(Enum):
    OSCILLATORS = auto()
    FILTER = auto()
    AMP_ENV = auto()
    MOD_ENV = auto()
    LFO = auto()
    MASTER = auto()


_MODULE_PARAM_NAMES: Dict[ActionModule, Sequence[str]] = {
    ActionModule.OSCILLATORS: (
        "osc1_wave_morph",
        "osc1_unison",
        "osc1_detune_cents",
        "osc1_level",
        "osc2_wave_morph",
        "osc2_unison",
        "osc2_detune_cents",
        "osc2_level",
        "osc2_semitone_offset",
        "noise_level",
    ),
    ActionModule.FILTER: ("filter_cutoff", "filter_resonance", "filter_env_amount"),
    ActionModule.AMP_ENV: ("amp_attack", "amp_decay", "amp_sustain", "amp_release"),
    ActionModule.MOD_ENV: ("mod_attack", "mod_decay", "mod_sustain", "mod_release"),
    ActionModule.LFO: ("lfo_rate", "lfo_amount", "lfo_target"),
    ActionModule.MASTER: ("master_volume",),
}

_MODULE_PARAM_INDICES: Dict[ActionModule, torch.Tensor] = {
    module: torch.tensor([_PARAM_INDEX[name] for name in names], dtype=torch.long)
    for module, names in _MODULE_PARAM_NAMES.items()
}


@dataclass(frozen=True)
class CurriculumStage:
    """Defines which modules are editable during a curriculum stage."""

    name: str
    enabled_modules: Sequence[ActionModule]

    def parameter_mask(self) -> torch.Tensor:
        mask = torch.zeros(_PARAM_DIM, dtype=torch.bool)
        for module in self.enabled_modules:
            idx = _MODULE_PARAM_INDICES[module]
            mask[idx] = True
        return mask


DEFAULT_CURRICULUM: Sequence[CurriculumStage] = (
    CurriculumStage("stage_1_osc", (ActionModule.OSCILLATORS, ActionModule.MASTER)),
    CurriculumStage(
        "stage_2_filter",
        (
            ActionModule.OSCILLATORS,
            ActionModule.FILTER,
            ActionModule.MASTER,
        ),
    ),
    CurriculumStage(
        "stage_3_amp_env",
        (
            ActionModule.OSCILLATORS,
            ActionModule.FILTER,
            ActionModule.AMP_ENV,
            ActionModule.MASTER,
        ),
    ),
    CurriculumStage(
        "stage_4_unison_mod",
        (
            ActionModule.OSCILLATORS,
            ActionModule.FILTER,
            ActionModule.AMP_ENV,
            ActionModule.MOD_ENV,
            ActionModule.MASTER,
        ),
    ),
    CurriculumStage(
        "stage_5_lfo",
        (
            ActionModule.OSCILLATORS,
            ActionModule.FILTER,
            ActionModule.AMP_ENV,
            ActionModule.MOD_ENV,
            ActionModule.LFO,
            ActionModule.MASTER,
        ),
    ),
)


def parameter_indices_for(module: ActionModule) -> torch.Tensor:
    """Return indices of parameters associated with a module."""
    return _MODULE_PARAM_INDICES[module]


def action_mask_for(stage: CurriculumStage) -> torch.Tensor:
    """Boolean mask over parameters indicating which can be edited."""
    return stage.parameter_mask()


def param_dim() -> int:
    return _PARAM_DIM


__all__ = [
    "ActionModule",
    "CurriculumStage",
    "DEFAULT_CURRICULUM",
    "parameter_indices_for",
    "action_mask_for",
    "param_dim",
]
