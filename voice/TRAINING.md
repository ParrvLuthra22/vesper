# Training a custom wake word ("wake up daddy's home")

Vesper's wake stage is [openWakeWord](https://github.com/dscripka/openWakeWord).
Stage 1 ships working out of the box with the bundled **`hey_jarvis`** model, so
nothing is blocked on this. This document is for training your **own** phrase.

> This is a **one-time, 4–8 hour GPU job**. Do it on a machine with an NVIDIA GPU
> (or Colab). The result is a single `.onnx` you drop into `voice/models/` and
> select via config — no code change, no rebuild.

## 0. Install training deps

```bash
pip install "openwakeword[training]" kokoro scipy
```

## 1. Record ~30 real samples of your voice

Your real voice + mic make the model generalize past synthetic speech.

```bash
python voice/scripts/record_wake_samples.py --phrase "wake up daddy's home" -n 30
# -> voice/training_data/positives_real/sample_XXX.wav
```

Vary tone, distance, and speed a little between takes.

## 2. Get negatives + background datasets

- **Negatives** — a large set of *clearly different* speech and sounds so the
  model learns what is **not** the wake word. Use openWakeWord's precomputed
  negative features, or general speech corpora (Common Voice, LibriSpeech).
  **Do NOT use similar-sounding phrases as negatives** — near-misses like "wake
  up daddy" or "daddy's home" push the decision boundary the wrong way and cause
  either constant false triggers or a model that never fires.
- **Background noise** — room tone, TV, music, traffic (e.g. AudioSet / FMA) for
  augmentation (reverb + mixing at varied SNR).

## 3. Synthesize positives + train

`train_wake_word.py` synthesizes many positives across Kokoro voices (with
pitch/speed jitter), mixes in your real recordings, augments with reverb +
background noise, computes openWakeWord features, trains, and exports ONNX:

```bash
python voice/scripts/train_wake_word.py \
    --phrase "wake up daddy's home" \
    --name wake_up_daddys_home \
    --real-dir voice/training_data/positives_real \
    --negatives-dir ~/data/openwakeword_negatives \
    --background-dir ~/data/audioset_noise \
    --n-per-voice 400 --steps 50000
# -> voice/models/wake_up_daddys_home.onnx
```

Tune on a held-out set: raise positives / augmentation if it misses your voice;
add more (dissimilar) negatives if it false-triggers.

## 4. Deploy — hot-swap by config

Drop the `.onnx` in `voice/models/` (the trainer does this) and point config at
it. **Switching wake models is a config change + restart, no code change.**

```yaml
# config/settings.yaml
voice:
  input:
    enabled: true
    wake_model: "wake_up_daddys_home"   # resolves to voice/models/wake_up_daddys_home.onnx
    wake_threshold: 0.5                  # raise to reduce false triggers
```

```bash
python -m voice.input   # now wakes on "wake up daddy's home"
```

`wake_model` accepts a bundled name (`hey_jarvis`), a bare name resolved under
`voice/models/`, or an explicit `.onnx` path.
