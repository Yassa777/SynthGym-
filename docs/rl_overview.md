# RL Environment Overview

This document captures the core ideas from the proposed reinforcement learning loop for preset imitation with the Serum-lite synthesizer. It distills the task requirements into the major subsystems we will implement.

## Environment Fundamentals
- **Task**: Match a target preset by iteratively editing synthesizer parameters.
- **Episode cadence**: Each action triggers a full render of a 0.5 s probe sequence. Episodes terminate when the perceptual loss drops below \( \varepsilon \) or after T = 50 steps.
- **Initialization**: Target preset sampled from a curriculum bucket. Agent starts at a neutral patch with occasional noisy warm starts near the target.

## Observations
- Concatenated feature vector built from:
  - Current log-mel spectrogram (64 × ~43 bins).
  - Cached target log-mel spectrogram.
  - Delta features: MFCC L1, spectral centroid Δ, LUFS Δ.
  - Current normalized parameter vector (dimension `d=25`).
- Optional augmentation: feature differences between current and best-so-far render for additional shaping.

## Actions
- Two-headed policy:
  1. Discrete module selector {OSC, Filter, Env, LFO} realised via categorical distribution.
  2. Continuous Δ vector for module-specific parameters; tanh-scaled and clamped within [−Δmax, Δmax].
- Parameter updates applied to normalized [0,1] vector and snapped for discrete choices (e.g., unison voices).

## Rewards
- Step reward equals negative perceptual loss:
  \[
  r_t = - \left( w_{mr} L_{MR-STFT} + w_{mfcc} L_{MFCC} + w_{cent} |\Delta c| + w_{loud} |\Delta \text{LUFS}| \right)
  \]
- Episodic bonus: \( +\beta \max(0, L^{best}_{t-1} - L_t) \ ).
- Anti-cheat: loudness/DC penalties, clip penalty, loudness matching via LUFS normalization.
- Optional episodic CLAP/VGGish cosine reward computed on best-so-far buffer.

## Curriculum & Scheduling
- Staged parameter exposure:
  1. Oscillator gains/mix only.
  2. +Filter cutoff/Q.
  3. +Amp ADSR.
  4. +Unison/detune.
  5. +LFO routing.
- Advance stage when median evaluation loss on held-out presets < threshold.

## Exploration & Stability
- Base agent: SAC with entropy target tuned for ~20–30% action stochasticity early on.
- Auxiliary losses: action smoothness penalty λ ||Δ||², optional critic regularization.
- Softclip render to prevent loudness exploits.

## Batching Strategy
- Vectorized environment with 32–64 parallel episodes.
- Render probes for all envs via batched Torch operations.
- Cache target features per episode; recompute current features each step (every action) or subsample if needed.

## Baselines & Evaluation
- Run CMA-ES for coarse search (2–5k evals) → behavior cloning warm-start for SAC.
- Evaluate on fixed preset sets per curriculum stage; log median loss, success rate, steps to \(\varepsilon\), wall-clock.
- Logging artifacts: A/B waveforms, spectrograms, parameter trajectories, per-term losses, match meter scalar.

## Key Subsystems to Implement
1. **Synth wrapper** *(already built)*: normalized parameter interface and renderer.
2. **Feature extractor**: log-mel, MR-STFT, MFCC, loudness, centroid; reusable for target/current audio.
3. **Perceptual loss module**: combine weighted losses, track best-so-far buffers, anti-cheat penalties.
4. **Observation builder**: assemble feature tensors + params per environment instance.
5. **Action application logic**: module routing, parameter clamps, curriculum-based masks.
6. **Vectorized Gymnasium environment**: handles curriculum, caching, logging, termination.
7. **Training harness**: SAC implementation, replay buffers, batching across vectorized envs.
8. **Evaluation/Logging suite**: metrics, audio export, analytics dashboards.

This overview is the reference point for the detailed implementation plan and decision records in companion documents.
