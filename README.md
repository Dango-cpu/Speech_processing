# faster-phowhisper

`faster-phowhisper` is a Streamlit application for Vietnamese speech recognition
and English translation. It supports offline file transcription and browser
microphone streaming, exposes the local Streamlit server through ngrok, and uses
PhoWhisper for ASR in every path.

## What the app uses

- Vietnamese ASR: PhoWhisper only, for both ASR backends.
- Faster backend: `faster-whisper` with a CTranslate2-converted PhoWhisper
  checkpoint.
- Compatibility backend: `transformers` with `WhisperForConditionalGeneration`
  and `WhisperProcessor`.
- Cascaded Encoder backend: a PyTorch inference path for checkpoints exported by
  `notebooks/train_cascaded_phowhisper_colab_kaggle.ipynb`.
- Translation: `Helsinki-NLP/opus-mt-vi-en` through `transformers`.

There is no direct Whisper translation mode. Every mode first produces a
Vietnamese transcript with PhoWhisper, then translates that text with
`opus-mt-vi-en`.

## Project Structure

```text
.
├── app.py
├── asr/
│   ├── __init__.py
│   ├── audio.py
│   ├── cascaded_encoder_backend.py
│   ├── common.py
│   ├── faster_whisper_backend.py
│   └── transformers_backend.py
├── pipeline.py
├── pyproject.toml
├── README.md
├── translate/
│   ├── __init__.py
│   └── opus_mt.py
└── tunnel.py
```

Older repository folders may still exist, but the Streamlit app above is the
entry point for this project.

## Install

Install `uv` if needed:

```bash
pip install uv
```

Install dependencies:

```bash
uv sync
```

The `pyproject.toml` pins the PyTorch source to the CUDA 12.8 PyTorch index:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

CPU fallback is automatic when CUDA is not available.

## Configure ngrok

Create an ngrok account, copy your authtoken, and set it before launching the
app.

PowerShell:

```powershell
$env:NGROK_AUTHTOKEN="your-ngrok-token"
```

macOS/Linux:

```bash
export NGROK_AUTHTOKEN="your-ngrok-token"
```

If `NGROK_AUTHTOKEN` is missing, the app still runs locally and shows a graceful
sidebar warning with setup instructions.

## Run

```bash
uv run streamlit run app.py
```

When `NGROK_AUTHTOKEN` is set, the app opens an ngrok tunnel for the Streamlit
port and prints the public URL in the terminal. The same URL appears in the
Streamlit sidebar.

## Run on Colab or Kaggle

Use:

```text
notebooks/run_faster_phowhisper_colab_kaggle.ipynb
```

The notebook can use the repository files already present in the runtime, or it
can clone a repository URL you provide in `GITHUB_REPO_URL`. It installs with
`uv sync`, reads `NGROK_AUTHTOKEN` and optional `HF_TOKEN` from Colab/Kaggle
secrets, opens an ngrok tunnel, sets `APP_PUBLIC_URL`, and starts:

```bash
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For hosted runtimes, upload or mount local checkpoint folders before selecting
checkpoint-backed modes in the app:

- Faster-Whisper: `checkpoints/phowhisper-base-ct2/model.bin`
- Cascaded Encoder: `checkpoints/cascaded_phowhisper_ckpt/step_*/cascaded_phowhisper.pt`

## Faster-Whisper PhoWhisper Checkpoints

The Faster-Whisper backend requires a CTranslate2-converted PhoWhisper model.
The app defaults to:

```text
checkpoints/phowhisper-base-ct2
```

If that directory is missing, or if it does not contain `model.bin`, the app
will explain that the checkpoint must be converted first.

One conversion path is:

```bash
uv run ct2-transformers-converter \
  --model vinai/PhoWhisper-base \
  --output_dir checkpoints/phowhisper-base-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

For CPU-focused deployment, use int8 quantization:

```bash
uv run ct2-transformers-converter \
  --model vinai/PhoWhisper-base \
  --output_dir checkpoints/phowhisper-base-ct2-int8 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8
```

Then select the local converted directory in the sidebar as the custom
PhoWhisper model/path while using the Faster-Whisper backend.

## Cascaded Encoder Checkpoints

The Cascaded Encoder backend prefers the training notebook artifact:

```text
notebooks/train_cascaded_phowhisper_colab_kaggle.ipynb
```

That notebook saves checkpoints like:

```text
cascaded_phowhisper_ckpt/
└── step_500/
    ├── cascaded_phowhisper.pt
    ├── processor/
    └── config/
```

Copy or train/export that folder under `checkpoints/`, for example:

```text
checkpoints/cascaded_phowhisper_ckpt/step_500/cascaded_phowhisper.pt
```

In the Streamlit sidebar, choose `Cascaded Encoder backend`. The default path is:

```text
checkpoints/cascaded_phowhisper_ckpt
```

If the root contains multiple `step_*` folders, the app automatically loads the
highest-numbered step. You may also point the custom model/path directly at a
specific `step_*` directory or at `cascaded_phowhisper.pt`.

This backend is a PyTorch research/compatibility path, not a CTranslate2
runtime. It reconstructs the notebook architecture from the checkpoint metadata,
loads the saved `model_state_dict`, decodes Vietnamese with a simple greedy
decoder, and then sends the transcript through `Helsinki-NLP/opus-mt-vi-en`.

## Mode Combinations

### Faster-Whisper backend + Offline mode

Upload a `wav`, `mp3`, or `m4a` file. The app transcribes the full file with a
CTranslate2-converted PhoWhisper checkpoint through `faster-whisper`, using
built-in VAD/chunking. Each Vietnamese segment is translated with
`Helsinki-NLP/opus-mt-vi-en`.

### Faster-Whisper backend + Streaming mode

The browser microphone is captured through `streamlit-webrtc`. Audio is kept in
a rolling buffer, finalized in short chunks, transcribed with PhoWhisper through
`faster-whisper`, and translated chunk by chunk with `opus-mt-vi-en`.

### Transformers backend + Offline mode

Upload a `wav`, `mp3`, or `m4a` file. The app loads PhoWhisper natively through
`transformers`, transcribes the full file as Vietnamese text, then translates
the Vietnamese output with `opus-mt-vi-en`.

### Transformers backend + Streaming mode

The browser microphone is captured through `streamlit-webrtc`. The app uses a
manual RMS/silence threshold and rolling buffer for chunk finalization, then
transcribes each chunk with PhoWhisper through `transformers` and translates
with `opus-mt-vi-en`.

### Cascaded Encoder backend + Offline mode

Upload a `wav`, `mp3`, or `m4a` file. The app loads the notebook-exported
`cascaded_phowhisper.pt`, reconstructs the cascaded encoder architecture, runs
Vietnamese ASR in PyTorch, then translates the Vietnamese output with
`opus-mt-vi-en`.

### Cascaded Encoder backend + Streaming mode

The browser microphone is captured through `streamlit-webrtc`. The app uses the
same manual rolling buffer and RMS/silence threshold as the Transformers
streaming path, then runs each finalized chunk through the notebook-exported
Cascaded Encoder checkpoint and translates with `opus-mt-vi-en`.

## Device and Precision

The sidebar exposes `auto`, `cuda`, and `cpu`.

- Faster-Whisper on CUDA defaults to `compute_type="float16"`.
- Faster-Whisper on CPU defaults to `compute_type="int8"`.
- Transformers on CUDA uses `torch.float16`.
- Transformers on CPU uses `torch.float32`.
- Cascaded Encoder on CUDA uses `torch.float16`.
- Cascaded Encoder on CPU uses `torch.float32`.

If CUDA is selected but unavailable, the app falls back to CPU and displays a
warning.

## Hugging Face Model Caching

PhoWhisper and `Helsinki-NLP/opus-mt-vi-en` are downloaded through Hugging Face
the first time they are selected. Streamlit caches loaded model objects with
`st.cache_resource`, so subsequent runs in the same process reuse the loaded
models.

Use `HF_HOME`, `TRANSFORMERS_CACHE`, or the default Hugging Face cache location
to control where model files are stored.

## Error Handling

The app handles:

- missing `NGROK_AUTHTOKEN`;
- missing microphone permission or stopped WebRTC stream;
- unsupported audio file formats;
- missing or unconverted CTranslate2 checkpoints for Faster-Whisper;
- CUDA selection when CUDA is unavailable;
- model download/load failures surfaced as Streamlit errors.

## Development Notes

The UI calls only `pipeline.run_offline` and `pipeline.run_streaming`. ASR
loading and inference live under `asr/`, and translation is isolated under
`translate/`. This keeps ASR backends, offline/streaming orchestration, and
translation independently testable.
