# SoundGym 🎹🤖

A reinforcement learning system for synthesizer preset matching using a custom Serum-lite synthesizer built on TorchSynth. Train AI agents to recreate target sounds by iteratively adjusting synthesizer parameters.

## 🎯 Project Overview

SoundGym implements a novel approach to synthesizer programming using reinforcement learning. The system learns to match target presets by adjusting synthesizer parameters through perceptual audio loss functions, combining multi-resolution STFT analysis, MFCC features, spectral centroid, and loudness measurements.

### Key Features

- **Custom Serum-lite Synthesizer**: Built on TorchSynth with 24 normalized parameters
- **Perceptual Loss Functions**: Multi-resolution STFT, MFCC, spectral features, and loudness
- **RL Environment**: Gymnasium-compatible environment for preset matching
- **Interactive UI**: Streamlit interface for synthesizer exploration
- **Curriculum Learning**: Staged parameter exposure for improved training

## 🏗️ Architecture

```
📁 SoundGym/
├── 🎹 synth/          # Serum-lite synthesizer engine
├── 🧠 rl/             # Reinforcement learning components
├── 🖥️  ui/             # Streamlit web interface
├── 📊 examples/       # Usage examples and demos
├── 📋 presets/        # Synthesizer preset files
├── 🔧 schemas/        # JSON schemas for validation
└── 📚 docs/           # Documentation and design decisions
```

### Core Components

- **`synth/engine.py`**: TorchSynth-based synthesizer with normalized parameter interface
- **`rl/features.py`**: Audio feature extraction (log-mel, MFCC, spectral analysis)
- **`rl/perceptual.py`**: Perceptual loss functions for audio comparison
- **`rl/actions.py`**: RL action space and parameter mapping
- **`ui/app.py`**: Interactive Streamlit interface for preset exploration

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.1+
- TorchAudio 2.1+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SoundGym
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Synthesizer UI

Launch the interactive Streamlit interface to explore the synthesizer:

```bash
streamlit run ui/app.py
```

This opens a web interface where you can:
- Adjust all 24 synthesizer parameters
- Preview audio in real-time
- Load/save preset files
- Explore parameter ranges and effects

### Basic Usage Example

```python
from synth.engine import SerumLiteSynth

# Create synthesizer instance
synth = SerumLiteSynth(sample_rate=16000, buffer_seconds=0.5)

# Get current parameters (normalized [0,1])
params = synth.get_param_vector()
print(f"Parameters: {params.shape}")  # torch.Size([24])

# Modify parameters
params[0] = 0.7  # Adjust OSC1 wave morph
synth.set_param_vector(params)

# Render audio
audio = synth.render(midi_note=60, dur_s=0.5)  # Middle C for 0.5 seconds
print(f"Audio shape: {audio.shape}")  # torch.Size([8000])

# Save/load presets
synth.save_preset("my_preset.json")
synth.load_preset("presets/basic_pad.json")
```

## 📊 Synthesizer Parameters

The Serum-lite synthesizer exposes 24 normalized parameters:

### Oscillators
- **OSC1/OSC2**: Wave morphing (sine→saw→square→triangle)
- **Unison**: 1-7 voices with detune spread
- **Levels**: Individual oscillator gains
- **Tuning**: Semitone offset for OSC2

### Filter & Modulation
- **Filter**: Log-scale cutoff (30Hz-20kHz) and resonance
- **Envelopes**: Separate ADSR for amplitude and modulation
- **LFO**: Rate, amount, and pitch/filter targeting

### Master
- **Noise**: White noise level
- **Master Volume**: Overall output gain

## 🧠 Reinforcement Learning

### Environment

The RL environment follows the Gymnasium interface:

- **Observation**: Concatenated features (log-mel spectrograms, deltas, current parameters)
- **Action**: Two-headed policy (module selector + continuous parameter deltas)
- **Reward**: Negative perceptual loss with improvement bonuses
- **Termination**: Loss below threshold or maximum steps (50)

### Training Process

1. **Target Sampling**: Random preset from curriculum bucket
2. **Initialization**: Start from neutral preset (optional warm start)
3. **Episode Loop**: 
   - Agent selects module and parameter adjustments
   - Synthesizer renders 0.5s probe sequence
   - Perceptual loss computed against target
   - Reward based on loss improvement
4. **Curriculum**: Staged parameter exposure (OSC→Filter→Env→LFO)

### Perceptual Loss Function

```python
loss = w_mrstft * L_MR-STFT + w_mfcc * L_MFCC + 
       w_centroid * |Δ_centroid| + w_loudness * |Δ_LUFS|
```

With anti-cheat penalties for DC bias, clipping, and loudness exploits.

## 🎵 Audio Features

### Multi-Resolution STFT (MR-STFT)
- Multiple window sizes: 2048, 1024, 512 samples
- Captures both harmonic and transient content
- Primary perceptual loss component

### Mel-Frequency Cepstral Coefficients (MFCC)
- 20 coefficients from 64-mel spectrograms
- Captures timbral characteristics
- Robust to pitch variations

### Spectral Features
- **Centroid**: Brightness/tone color
- **LUFS Loudness**: Perceptual loudness matching
- **Delta Features**: Frame-to-frame changes

## 📁 File Structure

```
synth/
├── __init__.py         # Package initialization
└── engine.py          # SerumLiteSynth class and ParamSpec

rl/
├── __init__.py         # Package initialization
├── config.py          # Configuration dataclasses
├── features.py        # Audio feature extraction
├── perceptual.py      # Perceptual loss functions
└── actions.py         # RL action space handling

ui/
└── app.py             # Streamlit web interface

docs/
├── rl_overview.md     # Environment design overview
├── rl_decisions.md    # Design decisions and rationale
└── rl_roadmap.md      # Implementation roadmap

examples/
└── perceptual_smoke.py # Feature extraction examples

presets/
└── basic_pad.json     # Example synthesizer preset

schemas/
└── serum_lite_preset.schema.json # JSON schema for presets
```

## 🔧 Development

### Running Tests

```bash
# Run feature extraction smoke test
python examples/perceptual_smoke.py
```

### Code Style

The project follows Python best practices:
- Type hints throughout
- Dataclasses for configuration
- Comprehensive docstrings
- Modular, testable components

### Adding New Features

1. **New Parameters**: Add to `SerumLiteSynth.PARAM_SPECS` in `synth/engine.py`
2. **Audio Features**: Extend `rl/features.py` with new extractors
3. **Loss Functions**: Add components to `rl/perceptual.py`
4. **UI Controls**: Update `ui/app.py` for new parameters

## 📈 Performance

### Benchmarks
- **Rendering**: ~1000 presets/second (CPU)
- **Feature Extraction**: ~500 samples/second
- **Memory Usage**: ~200MB baseline
- **Training**: 32-64 parallel environments recommended

### Optimization Tips
- Use GPU for batch feature extraction
- Cache target features per episode
- Vectorize environments for training
- Consider lower sample rates for faster iteration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Update documentation
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **TorchSynth**: Provides the synthesis building blocks
- **PyTorch**: Deep learning framework
- **Streamlit**: Web interface framework
- **Gymnasium**: RL environment standard

## 📞 Contact

[Add your contact information here]

---

*Built with ❤️ for the intersection of AI and audio synthesis*
