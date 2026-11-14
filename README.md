# Music Generation with GPT-2

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abinaya25138/midi-music-generation-gpt2/blob/main/music_generation.ipynb)

A GPT-2 language model trained from scratch to compose multi-track MIDI music. MIDI files are cut into 8-bar pieces and written out as text, and the model learns to continue that text one musical event at a time. While it trains, it composes a new piece at every evaluation and plays it back, so you can hear it go from noise to music.

I built this pipeline during my internship at Intel to generate Indian classical music. No MIDI dataset existed for this music, so we built one: we started from the [Thaat and Raga Forest (TRF) dataset](https://ieee-dataport.org/documents/thaat-and-raga-forest-trf-dataset) of Hindustani recordings, classified the recordings, and converted them into MIDI chunks by hand. That MIDI dataset isn't included here. The example run below uses public MIDI files to show the code working end to end.

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

To show the code working, I trained the notebook's pipeline on 3,000 songs from the public [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (Clean MIDI subset) on an Apple M4 laptop.

| | |
|---|---|
| Training pieces | 40,983 eight-bar pieces (36,884 for training, 1,000 held out for validation) |
| Vocabulary | 434 tokens |
| Model | GPT-2, 6 layers, 8 heads, 512-dim embeddings, 19.4M parameters, randomly initialized |
| Training | 3,000 steps, batch size 16 (about 1.3 epochs) |

![Training and validation loss](results/loss_curve.png)

Validation loss fell from 1.93 to 0.99 over 3,000 steps and stayed below the training loss throughout, so the model is generalizing to pieces it hasn't seen rather than memorizing.

![Piano roll of a generated piece](results/generated_piano_roll.png)

A piece the model composed from the prompt `PIECE_START TRACK_START`: a bass line with guitar phrases on top, across three instruments. At this stage the output has a steady pulse and plausible instrument parts, but it is repetitive; longer training would be the next step.

Generated samples are in [`results/samples/`](results/samples). Open them in GarageBand or any MIDI player. `early_step_500.mid` is from early in training, for comparison.

## Pitch bends (optional)

In Indian classical music, much of the expression lives between the notes: slides, dips and oscillations called gamakas. MIDI stores these as pitch bends. My first version of the tokenizer kept only note pitches and threw the bends away, so the model could only jump between fixed pitches, and its output lacked gamakas.

Setting `KEEP_PITCH_BENDS = True` keeps them:

- A `BEND` token records the bend amount whenever it changes during a note, in quarter-semitone steps (`BEND=-2` is half a semitone flat, `BEND=0` is back in tune).
- The timing grid becomes sixty-fourth notes, so short ornaments survive.
- Only the points where the rounded bend changes are kept. Audio-to-MIDI converters write bends very densely, and every extra token uses up the 512-token context.
- The decoder turns `BEND` tokens back into pitch bends for playback and the `.mid` output.

```
NOTE_ON=57 TIME_DELTA=15 BEND=-1 TIME_DELTA=1 BEND=-2 TIME_DELTA=1 BEND=-1 TIME_DELTA=2 BEND=0 TIME_DELTA=15 NOTE_OFF=57
```

That line is from a piece the model generated: it holds an A, dips half a semitone and comes back to pitch.

**Tested on the example data**, not on Indian classical music, since the dataset built at Intel isn't included:

- 1,833 of the 3,000 Lakh MIDI files contain pitch bends; 12,566 of the 40,983 pieces keep at least one `BEND` token, and bends are 0.6% of all tokens.
- Every piece decodes back with the same number of pitch bends it was encoded with.
- After 2,000 training steps (validation loss 1.13), 1 of 6 generated pieces used bends ([`pitch_bends_example.mid`](results/samples/pitch_bends_example.mid)). Bends are rare in Western pop, so the model learns them slowly; music where bends are central, like the Hindustani recordings this was built for, would give it far more to learn from.

## Run it

1. Download MIDI files to train on, for example the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (the Clean MIDI subset is a good size to start with), and upload the folder to Google Drive.
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

- **Thaat and Raga Forest (TRF) dataset**, the base for the Indian classical MIDI dataset built at Intel: S. Majumder and A. Bhattacharya, IEEE DataPort, 2024, [doi:10.21227/0xtb-fn23](https://doi.org/10.21227/0xtb-fn23), CC BY 4.0. Related paper: S. Majumder and A. Bhattacharya, "An automatic thaat and raga identification system using CNN-based models", *Innovations in Systems and Software Engineering*, 2023, [doi:10.1007/s11334-023-00541-2](https://doi.org/10.1007/s11334-023-00541-2).
- **Lakh MIDI Dataset**, used for the example run: C. Raffel, *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016. [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/)

## Tech stack

PyTorch, Hugging Face Transformers, Tokenizers and Datasets, miditoolkit, note_seq, FluidSynth, Weights & Biases (optional)
