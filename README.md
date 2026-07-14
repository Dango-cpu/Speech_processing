# faster-phowhisper

`faster-phowhisper` is a Streamlit application for Vietnamese speech recognition
with PhoWhisper. It supports offline audio files, browser microphone streaming,
GPU execution on local machines or Colab/Kaggle, and public sharing through
ngrok.

The project is ASR-only.

## What The App Uses

- Vietnamese ASR: PhoWhisper.
- Faster backend: `faster-whisper` with a CTranslate2-converted PhoWhisper
  checkpoint.
- Compatibility backend: `transformers` with `WhisperForConditionalGeneration`
  and `WhisperProcessor`.
- Cascaded Encoder backend: a PyTorch inference path for checkpoints exported by
  `notebooks/train_cascaded_phowhisper_colab_kaggle.ipynb`.
- UI/runtime: Streamlit, `streamlit-webrtc`, and ngrok.

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
├── tunnel.py
└── notebooks/
    ├── run_faster_phowhisper_colab_kaggle.ipynb
    └── train_cascaded_phowhisper_colab_kaggle.ipynb
```

Older repository folders may still exist, but `app.py` is the Streamlit entry
point for this app.

## Install

Install `uv` if needed:

```bash
pip install uv
```

Install dependencies:

```bash
uv sync
```

The `pyproject.toml` pins PyTorch to the CUDA 12.8 PyTorch index:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

CPU fallback is automatic when CUDA is unavailable.

## Configure Ngrok

Set your ngrok authtoken before launching:

PowerShell:

```powershell
$env:NGROK_AUTHTOKEN="your-ngrok-token"
```

macOS/Linux:

```bash
export NGROK_AUTHTOKEN="your-ngrok-token"
```

If `NGROK_AUTHTOKEN` is missing, the app still runs locally and shows setup
instructions in the sidebar.

## Run Locally

```bash
uv run streamlit run app.py
```

When `NGROK_AUTHTOKEN` is set, the app opens an ngrok tunnel and displays the
public URL in the sidebar.

## Streamlit Model Loading Pattern

The app follows the usual Streamlit pattern for NLP/ASR projects:

- model-loading functions are wrapped with `st.cache_resource`;
- the sidebar shows backend, model path, and checkpoint status;
- `Load selected ASR model` loads the selected model before inference;
- actual transcription also loads lazily, so the first run works even if the
  user skips manual preload;
- only the selected ASR backend is loaded, avoiding unnecessary memory use.

Useful environment defaults:

```bash
APP_DEFAULT_DEVICE=cuda
APP_DEFAULT_COMPUTE_TYPE=float16
APP_DEFAULT_ASR_BACKEND=transformers
```

## Run On Colab Or Kaggle

Use:

```text
notebooks/run_faster_phowhisper_colab_kaggle.ipynb
```

The notebook can use files already present in the runtime, or it can clone a
repository URL you provide in `GITHUB_REPO_URL`. It installs with `uv sync`,
checks CUDA availability, sets GPU defaults, reads `NGROK_AUTHTOKEN` and
optional `HF_TOKEN` from Colab/Kaggle secrets, opens an ngrok tunnel, sets
`APP_PUBLIC_URL`, and starts:

```bash
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For hosted runtimes, upload or mount local checkpoint folders before selecting
checkpoint-backed modes:

- Faster-Whisper: `checkpoints/phowhisper-base-ct2/model.bin`
- Cascaded Encoder:
  `checkpoints/cascaded_phowhisper_ckpt/step_*/cascaded_phowhisper.pt`

## Faster-Whisper PhoWhisper Checkpoints

The Faster-Whisper backend requires a CTranslate2-converted PhoWhisper model.
The app defaults to:

```text
checkpoints/phowhisper-base-ct2
```

If that directory is missing, or if it does not contain `model.bin`, the app
explains that the checkpoint must be converted first.

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

## Cascaded Encoder Checkpoints

The Cascaded Encoder backend expects the training notebook export:

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

The app can load a specific `step_*` directory, the `.pt` file directly, or a
root directory containing multiple `step_*` folders. If multiple steps exist,
the highest-numbered step is selected.

The Cascaded Encoder path is a PyTorch research/compatibility path. It is not a
CTranslate2 runtime.

## Modes

### Faster-Whisper + Offline

Upload a `wav`, `mp3`, or `m4a` file. The app transcribes the full file with a
CTranslate2-converted PhoWhisper checkpoint through `faster-whisper`, using its
VAD/chunking support.

### Faster-Whisper + Streaming

The browser microphone is captured through `streamlit-webrtc`. Audio is kept in
a rolling buffer, finalized in short chunks, and transcribed with PhoWhisper
through `faster-whisper`.

### Transformers + Offline

Upload a `wav`, `mp3`, or `m4a` file. The app loads PhoWhisper through
`transformers` and transcribes the full file as Vietnamese text.

### Transformers + Streaming

The browser microphone is captured through `streamlit-webrtc`. The app uses a
manual RMS/silence threshold and rolling buffer for chunk finalization, then
transcribes each chunk with PhoWhisper through `transformers`.

### Cascaded Encoder + Offline

Upload a `wav`, `mp3`, or `m4a` file. The app loads the notebook-exported
`cascaded_phowhisper.pt`, reconstructs the cascaded encoder architecture, and
runs Vietnamese ASR in PyTorch.

### Cascaded Encoder + Streaming

The browser microphone is captured through `streamlit-webrtc`. The app uses the
same rolling buffer and RMS/silence threshold as the Transformers streaming
path, then runs each finalized chunk through the Cascaded Encoder checkpoint.

## Device And Precision

The sidebar exposes `auto`, `cuda`, and `cpu`.

- Faster-Whisper on CUDA defaults to `compute_type="float16"`.
- Faster-Whisper on CPU defaults to `compute_type="int8"`.
- Transformers on CUDA uses `torch.float16`.
- Transformers on CPU uses `torch.float32`.
- Cascaded Encoder on CUDA uses `torch.float16`.
- Cascaded Encoder on CPU uses `torch.float32`.

If CUDA is selected but unavailable, the app warns and falls back to CPU.

## WebRTC Notes

The app passes an explicit WebRTC ICE configuration so it does not depend on
Hugging Face TURN credentials. By default it uses Google STUN:

```text
stun:stun.l.google.com:19302
```

For stricter networks, provide your own TURN server:

```bash
WEBRTC_TURN_URLS=turn:your-turn-server:3478
WEBRTC_TURN_USERNAME=your-user
WEBRTC_TURN_CREDENTIAL=your-password
```

## Error Handling

The app handles:

- missing `NGROK_AUTHTOKEN`;
- missing microphone permission or stopped WebRTC stream;
- unsupported audio file formats;
- missing or unconverted CTranslate2 checkpoints for Faster-Whisper;
- missing Cascaded Encoder notebook exports;
- CUDA selection when CUDA is unavailable;
- model download/load failures surfaced as Streamlit errors.

## Development Notes

The UI calls only `pipeline.run_offline` and `pipeline.run_streaming`. ASR
loading and inference live under `asr/`; streaming/offline orchestration lives
in `pipeline.py`.
