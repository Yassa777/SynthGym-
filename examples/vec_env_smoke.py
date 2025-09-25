"""Smoke test for the vectorized PresetImitation environment."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl import EnvConfig, FeatureExtractorConfig, PresetImitationVecEnv


def main() -> None:
    num_envs = 4
    env = PresetImitationVecEnv(
        num_envs,
        env_config=EnvConfig(max_steps=6, success_threshold=0.01),
        feature_config=FeatureExtractorConfig(),
    )

    obs, info = env.reset(seed=123)
    print("Observation batch shape:", obs.shape)
    print("Initial losses:", info["loss"])

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {step+1}: reward_mean={reward.mean():.4f}, loss_mean={info['loss'].mean():.4f},"
            f" terminated={terminated.any()}, truncated={truncated.any()}"
        )
    print("Final best losses:", info["best_loss"])


if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)
    main()
