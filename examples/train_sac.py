"""Example training script for SAC on the preset imitation environment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl import (
    ActionModule,
    EnvConfig,
    FeatureExtractorConfig,
    PerceptualLossConfig,
    ReplayBuffer,
    SacAgent,
    SacConfig,
    SacTrainer,
    make_env_fns,
)
from synth.engine import SerumLiteSynth


def evaluate_policy(agent: SacAgent, episodes: int = 2) -> Dict[str, float]:
    import gymnasium as gym

    env_config = EnvConfig(max_steps=50, success_threshold=0.01)
    env_fns = make_env_fns(episodes, env_config)
    env = gym.vector.SyncVectorEnv(env_fns, auto_reset=True)
    obs, _ = env.reset()
    cumulative = np.zeros(episodes, dtype=np.float32)
    for _ in range(env_config.max_steps):
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if np.all(np.logical_or(terminated, truncated)):
            break
    env.close()
    return {"mean_episode_reward": float(cumulative.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC agent for preset imitation.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    env_config = EnvConfig(max_steps=50, success_threshold=0.01)
    feature_config = FeatureExtractorConfig()
    loss_config = PerceptualLossConfig()

    env_fns = make_env_fns(args.num_envs, env_config, feature_config, loss_config)
    dummy_env = env_fns[0]()
    observation_dim = dummy_env.observation_space.shape[0]
    param_dim = len(SerumLiteSynth.PARAM_SPECS)
    dummy_env.close()

    device = torch.device(args.device)
    sac_config = SacConfig()
    agent = SacAgent(observation_dim, param_dim, sac_config, device=device)

    buffer = ReplayBuffer(args.buffer_size, observation_dim, param_dim, device=device, num_modules=len(ActionModule))

    trainer = SacTrainer(env_fns, agent, buffer, device=device)

    def eval_fn(agent: SacAgent) -> Dict[str, float]:
        try:
            return evaluate_policy(agent)
        except Exception:
            return {"mean_reward": 0.0}

    logs = trainer.train(args.total_steps, log_every=args.log_every, eval_fn=eval_fn)
    log_path = Path("sac_logs.json")
    log_path.write_text(json.dumps(logs, indent=2))
    print(f"Saved logs to {log_path}")


if __name__ == "__main__":
    main()
