"""Reinforcement learning toolkit for Serum-lite preset imitation."""

from .config import (
    EnvConfig,
    FeatureExtractorConfig,
    MRSTFTScale,
    PerceptualLossConfig,
    PerceptualLossWeights,
)
from .actions import ActionModule, CurriculumStage, DEFAULT_CURRICULUM
from .features import AudioFeatures, FeatureExtractor
from .observation import ObservationBuilder, ObservationSlices
from .perceptual import LossBreakdown, PerceptualEvaluator
from .env import PresetImitationEnv
from .vec_env import PresetImitationVecEnv
from .training import (
    ReplayBuffer,
    SacAgent,
    SacConfig,
    SacTrainer,
    CmaEsBaseline,
    CmaConfig,
)
from .training.sac import make_env_fns

__all__ = [
    "EnvConfig",
    "FeatureExtractorConfig",
    "MRSTFTScale",
    "PerceptualLossConfig",
    "PerceptualLossWeights",
    "ActionModule",
    "CurriculumStage",
    "DEFAULT_CURRICULUM",
    "AudioFeatures",
    "FeatureExtractor",
    "ObservationBuilder",
    "ObservationSlices",
    "LossBreakdown",
    "PerceptualEvaluator",
    "PresetImitationEnv",
    "PresetImitationVecEnv",
    "ReplayBuffer",
    "SacAgent",
    "SacConfig",
    "SacTrainer",
    "CmaEsBaseline",
    "CmaConfig",
    "make_env_fns",
]
