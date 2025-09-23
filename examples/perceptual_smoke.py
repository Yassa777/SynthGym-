"""Smoke test for the feature extractor and perceptual evaluator."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.config import FeatureExtractorConfig, PerceptualLossConfig  # noqa: E402
from rl.features import FeatureExtractor  # noqa: E402
from rl.perceptual import PerceptualEvaluator  # noqa: E402
from synth.engine import SerumLiteSynth  # noqa: E402


def random_preset_vector(num_params: int) -> torch.Tensor:
    return torch.rand(num_params)


def main() -> None:
    synth = SerumLiteSynth()
    param_dim = synth.get_param_vector().numel()

    target_params = random_preset_vector(param_dim)
    synth.set_param_vector(target_params)
    target_audio = synth.render(midi_note=60, dur_s=0.5).unsqueeze(0)

    current_params = target_params * 0.2  # intentionally far
    synth.set_param_vector(current_params)
    current_audio = synth.render(midi_note=60, dur_s=0.5).unsqueeze(0)

    extractor = FeatureExtractor(FeatureExtractorConfig())
    evaluator = PerceptualEvaluator(extractor, PerceptualLossConfig(improvement_bonus=0.1))

    evaluator.reset(target_audio)
    result = evaluator.evaluate(current_audio)

    print("Perceptual loss total:", result.total.item())
    print("Reward (negative loss with bonus):", result.reward.item())
    print("Improved best:", bool(result.improved.item()))
    print("Loss components:")
    for name, tensor in result.terms.items():
        print(f"  {name}: {tensor.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    main()
