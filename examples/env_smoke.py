"""Simple smoke test for the PresetImitationEnv."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl import EnvConfig, FeatureExtractorConfig, PerceptualLossConfig, PresetImitationEnv


def main() -> None:
    env = PresetImitationEnv(
        env_config=EnvConfig(max_steps=5, success_threshold=0.01),
        feature_config=FeatureExtractorConfig(),
        loss_config=PerceptualLossConfig(),
    )

    obs, info = env.reset(seed=0)
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)
    target_feat = env.perceptual.target_features
    if target_feat is not None:
        print("Target loudness:", target_feat.loudness.detach().cpu().numpy())

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {step+1}: reward={reward:.4f}, loss={info['loss']:.4f}, terminated={terminated}, truncated={truncated}"
        )
        print("  terms:", info["terms"])
        if terminated or truncated:
            break


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    main()
