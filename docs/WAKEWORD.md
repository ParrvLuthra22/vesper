# Wake word: "hey jarvis" now, "hey vesper" later

Vesper listens with [openWakeWord](https://github.com/dscripka/openWakeWord). The default model is the bundled **`hey_jarvis`**.
Saying "Hey Vesper" would be nicer, but a custom model is only worth having if it is **at least as reliable** and **no more
trigger-happy** than the bundled one. This page is how to train one, how to switch to it, and how to prove it is safe.

## The knobs (config/settings.yaml)

| key | default | meaning |
|---|---|---|
| `voice.wake.threshold` | `0.35` | score (0..1, one per 80 ms) at/above which a window counts |
| `voice.wake.patience_hits` / `patience_window` | `2` / `3` | fire only if this many of the last N windows count (a cough or door can't wake it) |
| `voice.wake.model_path` | `""` | path to a custom `.onnx` (e.g. `voice/models/hey_vesper.onnx`); empty = `voice.input.wake_model` |
| `voice.input.wake_model_fallback` | `hey_jarvis` | used if the custom file is missing, with a warning in the log |

`vesper doctor` shows `Wake model`: PASS when the file exists, WARN (with the fix) when `model_path` points at nothing.
Voice input logs the **max score per 10 s window** (scores only, never audio) at INFO whenever it is at least 0.05, so you can see
near-misses in `vesper logs voice_input`.

## 1. Train (Colab, ~1-2 h, GPU)

`scripts/train_hey_vesper/` — open `train_hey_vesper.ipynb` in Colab. It synthesizes thousands of "hey vesper" clips with Piper
voices, augments them with room noise, trains against ~2000 h of negative speech/noise, and exports `hey_vesper.onnx`.
Training was **not run** as part of writing this; expect to iterate on `CONFUSABLES` and `FALSE_POSITIVE_PENALTY`.

```bash
cp ~/Downloads/hey_vesper.onnx voice/models/hey_vesper.onnx
```

## 2. Try it side by side (does not change the default)

Edit config/settings.yaml: `voice.wake.model_path: "voice/models/hey_vesper.onnx"`, then:

```bash
vesper doctor                 # Wake model must PASS
vesper mic-test               # say "Hey Vesper" at normal speed, 5 times, a metre away: max score each time
vesper mic-test --selftest    # synthetic phrase through the same path (this tests the PIPELINE with the configured model)
```

Hit rate target: **at least 9 of 10** "Hey Vesper" at a normal pace (not slowly, not shouting) score above the threshold.

## 3. The false-accept test (required before switching the default)

A wake word that fires on the TV is worse than one that sometimes needs repeating.

1. Set `voice.wake.model_path` to the candidate and run voice input with the real microphone in the room.
2. Play **10 minutes of speech-heavy audio at normal volume** from speakers (not headphones), *without* saying the wake phrase:
   a podcast or news video, then a video with names in it (conversation, interviews). Sound-alike speech matters most: say
   "vesper", "hey Vesta", "hey Jasper", "hey there" a few times too.
3. Count wakes in that window: `vesper logs voice_input --tail 2000 | grep -c "idle -> wake_detected"` (counts only; add
   `--redact-transcripts` if you paste anything).
4. Read the near-misses: `vesper logs voice_input --tail 2000 | grep "wake: max score"` — the max per 10 s. If the highest max is
   above ~0.30 the model is close to firing on ordinary speech.

**Pass:** 0 wakes in the 10 minutes (and no 10 s max at or above the threshold). **Fail:** any wake → retrain with the
sound-alike phrase added to `CONFUSABLES` and/or a higher `FALSE_POSITIVE_PENALTY`, or raise `voice.wake.threshold`.
Repeat the same test with `hey_jarvis` first so you know the baseline of *your* room.

## 4. When it is safe to switch the default

All of these:

* hit rate ≥ 9/10 at a normal pace, in the room you actually use;
* the false-accept test passes **twice** (two different 10-minute videos), with the same threshold and patience as production;
* `vesper mic-test --selftest` still says `pipeline OK` with the new model (and the negative controls stay at 0);
* a day of normal use with `voice_input` running shows no unexplained `wake_detected` in the log.

Then change `voice.input.wake_model` (and keep `voice.wake.model_path` empty), keep `hey_jarvis` as `wake_model_fallback`, and
update the persona/greeting text if it mentions the old phrase. To go back: clear `voice.wake.model_path`.

## Why 0.35 + patience instead of 0.5

In the 2026-10-17 live test `hey_jarvis` needed to be said slowly and clearly at the 0.5 single-window threshold, although
synthetic "Hey Jarvis" scores 0.94-0.999 at every speaking speed (0.8x-1.6x). Two windows out of three at 0.35 is more forgiving
of a natural delivery and *less* sensitive to a single noisy window. The tests in `tests/test_wake_patience.py` run speech-like
noise and TTS of other phrases ("Hey Siri", "Hey Travis", ...) through the real model and require zero wakes.
