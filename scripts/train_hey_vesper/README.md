# train_hey_vesper

Colab-ready training of a custom **"hey vesper"** wake word with openWakeWord's synthetic pipeline and Piper voices.

* `train_hey_vesper.ipynb` — open in Google Colab (GPU runtime). `train_hey_vesper.py` is the same thing as `# %%` cells.
* Nothing here is run on the Mac, and **it has not been run at all yet** — it follows openWakeWord's upstream
  `automatic_model_training` notebook; if a step has changed upstream, the upstream notebook wins.
* Output: `hey_vesper.onnx` → `voice/models/hey_vesper.onnx`. Then read `docs/WAKEWORD.md` (switching, false-accept test).

Knobs that matter: `CONFUSABLES` (phrases that sound like "hey vesper" and must NOT fire), `FALSE_POSITIVE_PENALTY`
(raise it if the false-accept test fails), `N_SAMPLES`, `STEPS`.
