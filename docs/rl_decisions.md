# Decision Log & Open Questions

This living document records design decisions made while planning the RL pipeline alongside open issues that require future validation.

## Confirmed Decisions

### D1 — Observation Structure
- **Decision**: Observations will concatenate current log-mel spectrogram, cached target log-mel, auxiliary delta features, and the full normalized parameter vector.
- **Rationale**: Provides the agent with immediate perceptual context plus the actionable state (parameters). Maintaining normalized params ensures compatibility with CMA-ES and SAC.
- **Implication**: Observation dimensionality is high (~5k+). We'll implement feature flattening and consider lightweight CNN encoders later if MLPs struggle.

### D2 — Action Topology
- **Decision**: Two-head policy (module categorical + Δ vector) is the default action interface.
- **Rationale**: Encourages structured exploration; aligns with curriculum stages adding modules gradually.
- **Implication**: Action decoder needs curriculum-aware masking to avoid editing locked modules.

### D3 — Reward Composition
- **Decision**: Weighted blend of MR-STFT, MFCC L1, spectral centroid Δ, and loudness Δ, with anti-cheat penalties and episodic improvement bonus.
- **Rationale**: Balances spectral detail, timbre envelope, and level alignment. Improvement bonus stabilizes learning by rewarding progress.
- **Implication**: Feature stack must expose term-level breakdown for logging and tuning.

### D4 — Curriculum Advancement Criterion
- **Decision**: Promote to the next stage when median loss on held-out presets for the current stage drops below a threshold.
- **Rationale**: Maintains consistent difficulty ramp; avoids overfitting on training presets.
- **Implication**: Requires evaluation harness and stage-specific validation sets early in development.

### D5 — Baseline Strategy
- **Decision**: Run CMA-ES as a coarse search baseline and capture top trajectories for behavior cloning/SAC warm-start.
- **Rationale**: CMA handles high-dimensional continuous search well without gradients, providing a strong starting distribution.
- **Implication**: Need infrastructure to translate CMA trajectories into replay buffer seed data.

## Open Questions & Risks

### Q1 — Feature Dimensionality vs. Sample Efficiency
- **Issue**: 64 × 43 log-mel frames per observation may be heavy for SAC; flattening might limit learning.
- **Next Steps**: Prototype lightweight CNN or PCA compression; explore autoencoder pretraining.

### Q2 — Vectorized Rendering Performance
- **Issue**: TorchSynth-based rendering may bottleneck when scaling to 64 envs × 0.5 s audio.
- **Next Steps**: Benchmark batched rendering, investigate JIT/torch.compile or migrating to TorchScript/Accelerated CPU path.

### Q3 — CLAP/VGGish Integration Cost
- **Issue**: Episodic embedding scores may require GPU or large models.
- **Next Steps**: Determine feasible model (e.g., YAMNet, smaller CLAP variant) and compute budget; consider separate worker process.

### Q4 — Action Smoothing vs. Responsiveness
- **Issue**: λ ||Δ||² penalty aids stability but may slow convergence when large edits are needed.
- **Next Steps**: Experiment with adaptive λ based on stage or best-loss trend; possibly incorporate trust region constraints.

### Q5 — Logging Volume Management
- **Issue**: Saving audio A/B and spectrograms for every episode could bloat storage.
- **Next Steps**: Define retention policy (e.g., keep top-K per stage), compress audio, or integrate cloud storage hooks.

### Q6 — Discrete Parameter Handling
- **Issue**: Unison voice count and similar discrete parameters are currently normalized floats.
- **Next Steps**: Decide whether to treat as categorical actions or continue snapping post-update; evaluate impact on learning.

## To Monitor
- Validate loudness normalization to prevent reward gaming before integrating with agents.
- Verify curriculum thresholds using pilot runs; thresholds may need manual tuning.
- Ensure reproducibility in vectorized environment (seeding, noise) before large-scale training.

Update this log as decisions evolve or new questions emerge.
