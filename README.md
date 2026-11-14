# Music Generation with GPT-2

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abinaya25138/midi-music-generation-gpt2/blob/main/music_generation.ipynb)

A GPT-2 language model trained from scratch to compose multi-track MIDI music. MIDI files are cut into 8-bar pieces and written out as text, and the model learns to continue that text one musical event at a time. While it trains, it composes a new piece at every evaluation and plays it back, so you can hear it go from noise to music.

I built this pipeline during my internship at Intel to generate Indian classical music. No MIDI dataset existed for this music, so we built one: we started from the [Thaat and Raga Forest (TRF) dataset](https://ieee-dataport.org/documents/thaat-and-raga-forest-trf-dataset) of Hindustani recordings, classified the recordings, and converted them into MIDI chunks by hand. That MIDI dataset isn't included here. The example run below rebuilds a smaller version of it from two ragas of the same TRF recordings.

## How it works

1. **Cut into pieces.** Every MIDI file is split into 8-bar chunks (4/4). Notes go to the chunk they start in, with times shifted so each chunk starts at zero. Chunks with fewer than 10 notes are dropped.
2. **Write as text.** Each chunk is encoded in the MMM (Multi-Track Music Machine) token format. Tracks come one after another, each made of bars, each bar a list of note events:

   ```
   PIECE_START TRACK_START INST=0 DENSITY=2
     BAR_START NOTE_ON=60 TIME_DELTA=4 NOTE_OFF=60 TIME_DELTA=2 NOTE_ON=64 ... BAR_END
     ...
   TRACK_END TRACK_START INST=33 ... TRACK_END PIECE_END
   ```

   `INST` is the General MIDI instrument, and `TIME_DELTA` moves time forward in sixteenth notes. Snapping every file to the same sixteenth-note grid keeps the vocabulary small, whatever each file's original timing resolution.
3. **Tokenizer.** GPT-2's English tokenizer would break `NOTE_ON=60` into fragments, so a word-level tokenizer is trained instead: every distinct MMM token is one vocabulary entry. It splits on spaces only. Splitting on `=` as well would make the generated text impossible to decode back into notes.
4. **Model.** GPT-2 with 6 layers, 8 attention heads and 512-dimensional embeddings, trained from random weights on next-token prediction with a 512-token context.
5. **Back to music.** A decoder turns generated tokens into a `NoteSequence`, restarting the clock at each track and snapping to bar lines, and closes any notes the model left open. It's rendered to audio with FluidSynth and saved as `.wav` and `.mid`.

## Example run

I rebuilt a small Indian classical dataset from the TRF recordings and trained the notebook's pipeline on it, with pitch bends kept, on an Apple M4 laptop.

1. **Recordings:** all 41 TRF recordings of two ragas, Kedar (Kalyan thaat) and Bihag (Bilaval thaat), 4.0 hours in total.
2. **Audio to MIDI:** [`convert_audio_to_midi.py`](convert_audio_to_midi.py) runs Spotify's Basic Pitch with pitch bends on, so the gamakas survive as bends.
3. **Training:** the notebook's own chunking, encoding, tokenizer and model code with `KEEP_PITCH_BENDS = True`.

| | |
|---|---|
| Training pieces | 892 eight-bar pieces: 834 from 37 recordings for training, 58 from 4 held-out recordings (two per raga) for validation |
| Pitch bends | 98% of pieces contain `BEND` tokens |
| Vocabulary | 254 tokens, 13 of them bend levels |
| Model | GPT-2, 6 layers, 8 heads, 512-dim embeddings, 19.3M parameters, randomly initialized |
| Training | 2,000 steps, batch size 16; samples come from the checkpoint with the lowest validation loss (step 750) |

![Training and validation loss](results/loss_curve.png)

Validation pieces come from whole recordings the model never trained on, so it can't score well by remembering a performance. Validation loss bottomed out at 2.02 at step 750; after that, training loss kept falling while validation loss rose, the model memorizing 37 recordings rather than learning more about the ragas. Four hours of music is small for a model this size, which is why the samples come from step 750.

![Piano roll of a generated piece, showing pitch bends](results/generated_piano_roll.png)

A piece the model composed from the prompt `PIECE_START TRACK_START`. The shaded blocks are the notes as written; the line is the pitch you hear once the bends are applied. The model slides into notes and oscillates around them, the shapes of gamakas, rather than jumping between fixed pitches. Every one of the 8 pieces it composed uses bends. They are bent by half a semitone or more for 0% to 56% of their note time, within the range of the training recordings (4% to 73%), so the bends behave like the music's rather than leaving notes stuck out of tune.

Generated samples are in [`results/samples/`](results/samples). Open them in GarageBand or any MIDI player. They play on an electric piano, the instrument Basic Pitch assigns, not a sitar or voice. `early_step_500.mid` is from early in training, for comparison.

What this run does not show: the notes don't yet follow a raga's rules. A note-by-note comparison with the training recordings is weak, partly because each recording is in a different key. Learning raga grammar from four hours of music is the open problem; more recordings, transposing every recording to a common tonic, and training with the raga as a label are the next steps.

## Pitch bends (optional)

In Indian classical music, much of the expression lives between the notes: slides, dips and oscillations called gamakas. MIDI stores these as pitch bends. My first version of the tokenizer kept only note pitches and threw the bends away, so the model could only jump between fixed pitches, and its output lacked gamakas.

Setting `KEEP_PITCH_BENDS = True` keeps them:

- A `BEND` token records the bend amount whenever it changes during a note, in quarter-semitone steps (`BEND=-2` is half a semitone flat, `BEND=0` is back in tune).
- The timing grid becomes sixty-fourth notes, so short ornaments survive.
- Only the points where the rounded bend changes are kept. Audio-to-MIDI converters write bends very densely, and every extra token uses up the 512-token context.
- The decoder turns `BEND` tokens back into pitch bends for playback and the `.mid` output.

```
NOTE_ON=56 BEND=3 TIME_DELTA=1 BEND=1 TIME_DELTA=3 BEND=3 TIME_DELTA=2 BEND=1 TIME_DELTA=1 BEND=3 TIME_DELTA=1 BEND=1 TIME_DELTA=1 BEND=3 TIME_DELTA=2 NOTE_OFF=56
```

That line is from a piece the model generated in the example run: it holds a note and oscillates between a quarter and three quarters of a semitone above it, an andolan-like gamaka.

In the example run's raga recordings, bends are about 6% of all tokens and appear in 98% of pieces. Every piece decodes back with exactly the pitch bends it was encoded with. A check on 3,000 Western pop and rock MIDI files from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) found bends in only 31% of pieces and 0.6% of tokens, and a model trained on them rarely used bends at all: these tokens matter far more for Indian classical music.

## Run it

1. Get MIDI files to train on and upload the folder to Google Drive. To build Indian classical MIDI like the example run, download TRF recordings (from [Kaggle](https://www.kaggle.com/datasets/suryamajumder/thaat-and-raga-forest-trf-dataset) or IEEE DataPort) and convert them with `python convert_audio_to_midi.py AUDIO_DIR MIDI_DIR`, then set `KEEP_PITCH_BENDS = True`. Any other MIDI collection works too, such as the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).
2. Open the notebook in Colab with the badge above and choose a GPU runtime.
3. Set `MIDI_DIR` in the settings cell to that Drive folder, then run all cells.

Optional:

- **Weights & Biases logging.** Add your API key as a Colab secret named `WANDB_API_KEY`.
- **Publish the dataset and tokenizer to the Hugging Face Hub.** Set `HF_USERNAME`.

Everything the notebook produces is saved to `MyDrive/music_gpt2/`:

| Path | What it is |
| --- | --- |
| `pieces.txt` | The tokenized dataset, one 8-bar piece per line |
| `tokenizer/` | The trained word-level tokenizer |
| `checkpoints/`, `model/` | Training checkpoints and the final model |
| `generated/` | Pieces composed during and after training, as `.wav` and `.mid` |

## Limitations

- Each track is treated as one note at a time, so chords inside a track are spread out into short runs of notes.
- Pieces are assumed to be in 4/4 and are played back at 120 BPM, because the tokens don't record tempo.
- Drum tracks aren't marked separately, so they play back on a melodic instrument.
- Pitch bends are off by default. When on, they assume the standard ±2-semitone bend range, and a bend the model never resets leaves the following notes out of tune.
- Busy multi-track pieces longer than 512 tokens are truncated, which drops their last tracks.

## Data credits

- **Thaat and Raga Forest (TRF) dataset**, the base for the Indian classical MIDI dataset built at Intel and for the example run: S. Majumder and A. Bhattacharya, IEEE DataPort, 2024, [doi:10.21227/0xtb-fn23](https://doi.org/10.21227/0xtb-fn23), CC BY 4.0; also on [Kaggle](https://www.kaggle.com/datasets/suryamajumder/thaat-and-raga-forest-trf-dataset), where the example run's recordings came from. Related paper: S. Majumder and A. Bhattacharya, "An automatic thaat and raga identification system using CNN-based models", *Innovations in Systems and Software Engineering*, 2023, [doi:10.1007/s11334-023-00541-2](https://doi.org/10.1007/s11334-023-00541-2).
- **Basic Pitch**, used to convert the recordings to MIDI: R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal and S. Ewert, "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022. [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch)
- **Lakh MIDI Dataset**, used for the pitch-bend comparison: C. Raffel, *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016. [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/)

## Tech stack

PyTorch, Hugging Face Transformers, Tokenizers and Datasets, miditoolkit, note_seq, FluidSynth, Basic Pitch, Weights & Biases (optional)
