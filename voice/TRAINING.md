# Training a custom wake word ("wake up daddy's home")

Vesper's wake stage is [openWakeWord](https://github.com/dscripka/openWakeWord).
Stage 1 ships working out of the box with the bundled **`hey_jarvis`** model, so
nothing is blocked on this. This document is for training your **own** phrase.

> This is a **one-time, 4–8 hour GPU job**. Do it on a machine with an NVIDIA GPU
> (or Colab). The result is a single `.onnx` you drop into `voice/models/` and
> select via config — no code change, no rebuild.

## Status

**Not yet trained.** `voice/models/wake_up_daddys_home.onnx` does not exist, so
`voice.input.wake_model` is set to `hey_jarvis` and the pipeline is verified
end-to-end on that (wake score 0.997 against a 0.5 threshold). Two things are
required that only you can supply: **~30 recordings of your own voice**, and a
**GPU for the 4–8 hour training run**. Everything else — synthesis, the
trainer, the config swap, the fallback — is in place and tested.

When the `.onnx` lands in `voice/models/`, it is a one-line config change
(step 4). Until then `wake_model_fallback` keeps voice input working.

## 0. Install training deps

```bash
pip install "openwakeword[training]" scipy
```

Positives are synthesized with **kokoro-onnx**, which Vesper already uses for
speech output — so if `voice/models/kokoro-v1.0.onnx` is present (it is), you
need no extra TTS install and no second model download. The trainer falls back
to the PyTorch `kokoro` package only if kokoro-onnx is unavailable.

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

The synthesis step alone is quick and CPU-only — worth running locally first to
confirm the phrase sounds right across voices before committing to a GPU run:

```bash
python - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("tw","voice/scripts/train_wake_word.py")
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
print(tw._synthesize_kokoro_onnx("wake up daddy's home", "/tmp/pos", n_per_voice=1), "clips")
EOF
```

Tune on a held-out set: raise positives / augmentation if it misses your voice;
add more (dissimilar) negatives if it false-triggers.

## 3b. No local GPU? Train it free on Colab or Kaggle

This Mac has no NVIDIA GPU, so the training run belongs on a free hosted one.
Colab gives ~T4-class GPUs; Kaggle gives ~30 GPU-hours/week and longer sessions.
Either is comfortably enough for a single wake word.

1. **Record locally first** (step 1) — it needs *your* microphone and voice.
   Zip the result:
   ```bash
   zip -r positives_real.zip voice/training_data/positives_real
   ```

2. **New Colab notebook** → Runtime → Change runtime type → **GPU (T4)**.

3. **Setup cell:**
   ```python
   !pip -q install "openwakeword[training]" kokoro scipy soundfile
   !git clone https://github.com/ParrvLuthra22/vesper.git && cd vesper
   from google.colab import files; files.upload()      # positives_real.zip
   !unzip -q positives_real.zip -d vesper/
   ```

4. **Datasets cell** — negatives and background noise (the two big downloads;
   openWakeWord publishes precomputed negative features, which is much faster
   than pulling raw corpora):
   ```python
   !mkdir -p /content/neg /content/bg
   !wget -q https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy -P /content/neg
   !wget -q https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train00.tar -P /content/bg
   !cd /content/bg && tar -xf bal_train00.tar
   ```

5. **Train cell** (the 4–8 hour part — keep the tab alive):
   ```python
   !cd vesper && python voice/scripts/train_wake_word.py \
       --phrase "wake up daddy's home" --name wake_up_daddys_home \
       --real-dir voice/training_data/positives_real \
       --negatives-dir /content/neg --background-dir /content/bg \
       --n-per-voice 400 --steps 50000
   ```

6. **Download the result:**
   ```python
   from google.colab import files
   files.download("vesper/voice/models/wake_up_daddys_home.onnx")
   ```

7. Drop it into `voice/models/` on this machine and do step 4 below.

> Colab free tier disconnects on idle. Keep the browser tab open, and prefer
> Kaggle if you need a session longer than ~4 hours.

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
