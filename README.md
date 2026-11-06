# Music Generation with GPT-2

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abinaya25138/midi-music-generation-gpt2/blob/main/music_generation.ipynb)

A GPT-2 language model trained from scratch to compose multi-track MIDI music. MIDI files are cut into 8-bar pieces and written out as text, and the model learns to continue that text one musical event at a time. While it trains, it composes a new piece at every evaluation and plays it back, so you can hear it go from noise to music.

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
- Busy multi-track pieces longer than 512 tokens are truncated, which drops their last tracks.

## Tech stack

PyTorch, Hugging Face Transformers, Tokenizers and Datasets, miditoolkit, note_seq, FluidSynth, Weights & Biases (optional)
