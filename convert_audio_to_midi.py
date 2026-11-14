"""Convert recordings to MIDI with Spotify's Basic Pitch, keeping pitch bends.

This is how the example run's MIDI was made from TRF recordings. Each subfolder of
AUDIO_DIR (one per raga) becomes a subfolder of MIDI_DIR.

    pip install basic-pitch
    python convert_audio_to_midi.py AUDIO_DIR MIDI_DIR
"""
import sys
from pathlib import Path

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

audio_dir, midi_dir = Path(sys.argv[1]), Path(sys.argv[2])


def load_model():
    """Basic Pitch ships the same model in several formats; use the first one this machine can run."""
    folder = Path(ICASSP_2022_MODEL_PATH).parent
    for candidate in (ICASSP_2022_MODEL_PATH, folder / "nmp.mlpackage", folder / "nmp.tflite", folder / "nmp.onnx"):
        try:
            return Model(candidate)
        except Exception:
            continue
    raise RuntimeError("no Basic Pitch model format could be loaded; install tensorflow, coremltools or onnxruntime")


model = load_model()

paths = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac"))
for k, path in enumerate(paths, 1):
    out = midi_dir / path.relative_to(audio_dir).with_suffix(".mid")
    if out.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    _, midi, _ = predict(str(path), model)  # pitch bends are written by default
    midi.write(str(out))
    bends = sum(len(i.pitch_bends) for i in midi.instruments)
    print(f"[{k}/{len(paths)}] {out.relative_to(midi_dir)}: {midi.get_end_time() / 60:.1f} min, {bends} pitch bends")
