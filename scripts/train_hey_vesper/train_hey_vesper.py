# %% [markdown]
# # Train `hey_vesper.onnx` (openWakeWord synthetic training, Piper voices)
#
# **Run this in Google Colab with a GPU runtime** (Runtime > Change runtime type > T4). It is NOT run on the Mac and has NOT
# been run by the author of this file: it follows openWakeWord's published "automatic model training" flow
# (https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb). If a step has moved upstream,
# the upstream notebook wins; the config values below are the ones that matter for Vesper.
#
# Output: `hey_vesper.onnx`. Copy it to `voice/models/hey_vesper.onnx`, set `voice.wake.model_path`, then follow
# `docs/WAKEWORD.md` (false-accept test) BEFORE making it the default.
#
# Plan: ~1-2 h on a T4. Downloads ~10 GB (negative-feature set + room impulse responses + background audio).

# %% Install
# !pip -q install "openwakeword" piper-tts webrtcvad-wheels pyyaml scipy datasets soundfile torch torchaudio
# !git clone https://github.com/dscripka/openWakeWord.git && pip -q install -e ./openWakeWord
# !git clone https://github.com/rhasspy/piper-sample-generator.git
# !wget -q -O piper-sample-generator/models/en_US-libritts_r-medium.pt \
#       https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt

# %% Settings
import os, yaml

TARGET_PHRASE = "hey vesper"
# Phonetic spellings help Piper say it the way you do ("VES-per", not "vez-per"). Add what you hear Kokoro/Piper get wrong.
TARGET_PHRASES = ["hey vesper", "hey, vesper", "hey vessper"]
# Phrases that SOUND like it must be negatives, or the model fires on them. These matter more than anything else.
CONFUSABLES = ["hey vesta", "hey jasper", "hey westerner", "hey besser", "hey vest", "hey best for", "hey whisper",
               "vesper", "hey there", "hey siri", "hey jarvis", "hey google", "okay vesper not", "hey vespa",
               "hey vester", "hey fester", "hey vessel", "hey wesley"]
N_SAMPLES = 20000          # synthetic positives (Piper multi-speaker voices, varied speed/noise)
N_SAMPLES_VAL = 2000
STEPS = 50000              # training steps; more = fewer misses, slightly more false accepts
FALSE_POSITIVE_PENALTY = 100   # push this UP if the false-accept test fails (docs/WAKEWORD.md)
TARGET_FP_PER_HOUR = 0.2

config = {
    "model_name": "hey_vesper",
    "target_phrase": TARGET_PHRASES,
    "custom_negative_phrases": CONFUSABLES,
    "n_samples": N_SAMPLES,
    "n_samples_val": N_SAMPLES_VAL,
    "tts_batch_size": 50,
    "augmentation_batch_size": 16,
    "piper_sample_generator_path": "./piper-sample-generator",
    "output_dir": "./my_custom_model",
    "rir_paths": ["./mit_rirs"],
    "background_paths": ["./audioset_16k", "./fma"],
    "background_paths_duplication_rate": [1, 1],
    "false_positive_validation_data_path": "validation_set_features.npy",
    "augmentation_rounds": 1,
    "feature_data_files": {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"},
    "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
    "model_type": "dnn",
    "layer_size": 32,
    "steps": STEPS,
    "max_negative_weight": FALSE_POSITIVE_PENALTY,
    "target_false_positives_per_hour": TARGET_FP_PER_HOUR,
}
with open("hey_vesper.yaml", "w") as f:
    yaml.dump(config, f)

# %% Data (negatives + augmentation audio) - same sources as the upstream notebook
# !wget -q https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
# !wget -q https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
# MIT room impulse responses, AudioSet + FMA background clips: see "Download Data" in the upstream notebook (datasets library).

# %% Train: generate synthetic clips -> augment -> train (each is a separate stage in openwakeword/train.py)
# !python openWakeWord/openwakeword/train.py --training_config hey_vesper.yaml --generate_clips
# !python openWakeWord/openwakeword/train.py --training_config hey_vesper.yaml --augment_clips
# !python openWakeWord/openwakeword/train.py --training_config hey_vesper.yaml --train_model

# %% Export / download
# The trained model is ./my_custom_model/hey_vesper.onnx
# from google.colab import files; files.download("my_custom_model/hey_vesper.onnx")

# %% [markdown]
# ## After download (on the Mac)
# 1. `cp ~/Downloads/hey_vesper.onnx voice/models/hey_vesper.onnx`
# 2. `voice.wake.model_path: "voice/models/hey_vesper.onnx"` in config/settings.yaml, then `vesper doctor` (must PASS "Wake model").
# 3. Do the false-accept test in docs/WAKEWORD.md. Do not change the default until it passes.
