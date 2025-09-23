# RL Implementation Roadmap

This roadmap sequences the work required to stand up the preset-imitation RL system. Each milestone groups tasks that can be developed and validated together. Dependencies between milestones are noted to help with prioritization and parallelization where possible.

## Milestone 0 — Environment Foundations *(completed)*
- ✅ Serum-lite synth wrapper with normalized parameter interface.
- ✅ JSON schema + preset tooling.
- ✅ Streamlit UI for manual auditioning.

## Milestone 1 — Audio Feature Stack
**Goal:** Create reusable modules for feature computation and perceptual loss.
- Implement batched renderer helper that accepts lists of MIDI probes and durations.
- Build feature extraction pipelines (log-mel, MR-STFT, MFCC, spectral centroid, loudness) using PyTorch + torchaudio.
- Cache target features; design API for single-call “analyze audio → feature dict”.
- Implement perceptual loss aggregator with weighting configuration, anti-cheat penalties, and best-so-far tracking.
- Unit-test individual metrics against reference values (e.g., librosa).

## Milestone 2 — Observation & Action Interfaces
**Goal:** Define the Gym-compatible environment interface for a single episode.
- Observation builder that concatenates feature tensors and current params into flattened vector; include shape metadata for agents.
- Module-specific action decoding (categorical selector + Δ head). Apply curriculum masks to restrict deltas.
- Parameter update utilities: clamp, snap discrete values, track history for logging.
- Implement early-termination checks and best-loss bookkeeping.
- Write deterministic tests to ensure action application respects bounds and curriculum gating.

## Milestone 3 — Vectorized Environment & Curriculum
**Goal:** Scale the single-episode logic to batch execution.
- Wrap environment using Gymnasium vector API (e.g., `SyncVectorEnv` or custom with PyTorch operations).
- Implement curriculum manager controlling preset sampling, parameter exposure masks, and stage promotion criteria.
- Add support for parallel probe rendering and batched feature computation.
- Integrate episodic CLAP/VGGish scoring as optional callback (computed on CPU or separate worker to avoid blocking).
- Logging hooks capturing per-step loss components and audio snapshots.

## Milestone 4 — Training Harness & Baselines
**Goal:** Trainably integrate SAC and baseline optimizers.
- Set up replay buffer, SAC actor/critic networks sized for observation/action dims.
- Implement entropy target auto-tuning and action smoothing penalty λ ||Δ||².
- Integrate vectorized env with PyTorch training loop (device placement, gradient steps per env step).
- Implement CMA-ES baseline runner with logging + dataset capture for behavior cloning warm-start.
- Provide scripts/config for staged curriculum training (advancing when thresholds met).

## Milestone 5 — Evaluation & Experiment Tracking
**Goal:** Ensure reproducible experiments and rich diagnostics.
- Fixed evaluation preset suites per curriculum stage; evaluation loop computing success metrics.
- Logging outputs: TensorBoard/W&B metrics, saved audio A/B, spectrogram plots, parameter trajectories.
- CLI tooling for launching training, evaluation-only runs, and ablations (no CMA, no Δ actions, etc.).
- Documentation of metrics definitions and recommended hyperparameters.

## Milestone 6 — UX & Automation Enhancements *(stretch)*
- Integrate Streamlit UI into evaluation dashboard for quick A/B listening of saved episodes.
- Optional web-based library of presets with metadata/tags.
- Batch export of best-matching audio for marketing/demo purposes.

## Dependencies & Parallelization Notes
- Milestone 1 is prerequisite for accurate rewards; must be finalized before Milestone 2 completes.
- Milestones 2 and 3 can progress in parallel once feature stack API is stable (vectorization relies on observation builder signatures).
- Milestone 4 depends on stable env interface but CMA-ES baseline can start earlier using direct param vectors.
- Milestone 5 depends on logging hooks created in Milestones 3–4; can be fleshed out incrementally.
- Stretch tasks in Milestone 6 should not block RL loop readiness.

Keep this roadmap updated with actual progress, adjusting milestones as new insights appear during implementation.
