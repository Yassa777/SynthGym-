"""Minimal Streamlit UI for interacting with the Serum-lite synthesizer."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import streamlit as st
from jsonschema import ValidationError, validate

from synth.engine import SerumLiteSynth

SCHEMA_PATH = (Path(__file__).resolve().parent.parent / "schemas/serum_lite_preset.schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text())


def _default_preset() -> Dict[str, float]:
    synth = SerumLiteSynth()
    return synth.to_preset_dict()


@st.cache_resource
def _get_schema() -> Dict:
    return SCHEMA


def main() -> None:
    st.set_page_config(page_title="Serum-lite", layout="wide")
    st.title("Serum-lite Playground")
    st.caption("Adjust the normalized knobs and preview the synthesizer output.")

    schema = _get_schema()

    if "preset" not in st.session_state:
        st.session_state["preset"] = _default_preset()

    synth = SerumLiteSynth()

    with st.sidebar:
        st.subheader("Preset management")
        uploaded = st.file_uploader("Load preset JSON", type="json")
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read().decode("utf-8"))
                validate(instance=data, schema=schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                st.error(f"Failed to load preset: {exc}")
            else:
                st.session_state["preset"] = data
                st.success("Preset loaded.")
        if st.button("Reset to defaults"):
            st.session_state["preset"] = _default_preset()
        if st.button("Randomize"):
            st.session_state["preset"] = {
                key: float(np.clip(np.random.rand(), 0.0, 1.0))
                for key in st.session_state["preset"].keys()
            }
        st.download_button(
            "Download preset",
            data=json.dumps(st.session_state["preset"], indent=2).encode("utf-8"),
            file_name="serum_lite_preset.json",
            mime="application/json",
        )

    midi_note = st.slider("Preview MIDI note", min_value=21, max_value=108, value=60)
    duration = st.slider("Preview duration (seconds)", 0.1, float(synth.buffer_seconds), value=0.5)

    preset = st.session_state["preset"]

    def knob(label: str, key: str, step: float = 0.01) -> None:
        preset[key] = float(st.slider(label, 0.0, 1.0, float(preset[key]), step=step, key=key))

    osc_cols = st.columns(3)

    with osc_cols[0]:
        st.subheader("Oscillator 1")
        knob("Wave morph", "osc1_wave_morph")
        knob("Unison voices", "osc1_unison")
        knob("Detune", "osc1_detune_cents")
        knob("Level", "osc1_level")

    with osc_cols[1]:
        st.subheader("Oscillator 2")
        knob("Wave morph", "osc2_wave_morph")
        knob("Unison voices", "osc2_unison")
        knob("Detune", "osc2_detune_cents")
        knob("Level", "osc2_level")
        knob("Semitone offset", "osc2_semitone_offset")

    with osc_cols[2]:
        st.subheader("Noise & Master")
        knob("Noise level", "noise_level")
        knob("Master volume", "master_volume")

    st.divider()
    filt_cols = st.columns(3)

    with filt_cols[0]:
        st.subheader("Filter")
        knob("Cutoff", "filter_cutoff")
        knob("Resonance", "filter_resonance")
        knob("Env amount", "filter_env_amount")

    with filt_cols[1]:
        st.subheader("Amp Envelope")
        knob("Attack", "amp_attack")
        knob("Decay", "amp_decay")
        knob("Sustain", "amp_sustain")
        knob("Release", "amp_release")

    with filt_cols[2]:
        st.subheader("Mod Envelope & LFO")
        knob("Mod attack", "mod_attack")
        knob("Mod decay", "mod_decay")
        knob("Mod sustain", "mod_sustain")
        knob("Mod release", "mod_release")
        knob("LFO rate", "lfo_rate")
        knob("LFO amount", "lfo_amount")
        knob("LFO target", "lfo_target")

    param_vector = [preset[spec.name] for spec in synth.PARAM_SPECS]
    synth.set_param_vector(param_vector)
    denorm_preview = synth.get_denormalized_params()

    with st.expander("Denormalized parameter values"):
        rows = [
            {
                "parameter": name,
                "value": f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}",
            }
            for name, val in denorm_preview.items()
        ]
        st.table(rows)

    if st.button("Render preview", type="primary"):
        audio = synth.render(midi_note=midi_note, dur_s=duration).numpy()
        audio /= np.max(np.abs(audio) + 1e-6)
        buffer = io.BytesIO()
        sf.write(buffer, audio, synth.sample_rate, format="WAV")
        buffer.seek(0)
        st.audio(buffer.getvalue(), format="audio/wav")


if __name__ == "__main__":
    main()
