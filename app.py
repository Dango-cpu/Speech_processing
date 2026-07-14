from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from asr.audio import save_uploaded_audio
from pipeline import StreamingState, run_offline, run_streaming
from tunnel import start_ngrok_tunnel


DEFAULT_HF_MODEL = "vinai/PhoWhisper-base"
DEFAULT_CASCADE_ROOT = "checkpoints/cascaded_phowhisper_ckpt"
PHOWHISPER_MODELS = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "custom": "",
}


def default_ct2_model_path(model_size: str) -> str:
    if model_size == "custom":
        return "checkpoints/phowhisper-base-ct2"
    return f"checkpoints/phowhisper-{model_size}-ct2"


def is_probably_local_path(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path)
    normalized = model_name_or_path.replace("\\", "/")
    return (
        path.exists()
        or path.is_absolute()
        or "\\" in model_name_or_path
        or normalized.startswith(("./", "../", "checkpoints/"))
    )


def local_checkpoint_hint(asr_backend: str, model_name_or_path: str) -> str | None:
    path = Path(model_name_or_path)
    if (
        asr_backend == "faster_whisper"
        and is_probably_local_path(model_name_or_path)
        and not (path / "model.bin").exists()
    ):
        return (
            "Faster-Whisper needs a CTranslate2 PhoWhisper checkpoint with model.bin. "
            "Use the README conversion command if this path has not been converted yet."
        )
    if asr_backend == "cascaded_encoder":
        has_direct = path.is_file() and path.name == "cascaded_phowhisper.pt"
        has_step = (path / "cascaded_phowhisper.pt").exists()
        has_root = any(path.glob("step_*/cascaded_phowhisper.pt")) if path.exists() else False
        if not (has_direct or has_step or has_root):
            return (
                "Cascaded Encoder needs the notebook export: "
                "cascaded_phowhisper.pt inside a step_* folder or selected directly."
            )
    return None


st.set_page_config(
    page_title="faster-phowhisper",
    layout="wide",
)


def init_state() -> None:
    st.session_state.setdefault("stream_state", StreamingState())
    st.session_state.setdefault("stream_result", {"segments": [], "vi_text": "", "en_text": ""})


def sidebar_controls() -> dict[str, object]:
    st.sidebar.title("faster-phowhisper")
    port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    public_url, tunnel_error = start_ngrok_tunnel(port)
    if public_url:
        st.sidebar.success("ngrok tunnel active")
        st.sidebar.link_button("Open public URL", public_url)
        st.sidebar.caption(public_url)
    else:
        st.sidebar.warning(tunnel_error)

    asr_backend_label = st.sidebar.radio(
        "ASR backend strategy",
        [
            "Faster-Whisper backend",
            "Transformers backend",
            "Cascaded Encoder backend",
        ],
        help=(
            "All choices produce Vietnamese ASR, then opus-mt-vi-en performs "
            "English translation. Cascaded Encoder loads the notebook-exported "
            "PyTorch checkpoint."
        ),
    )
    if asr_backend_label == "Faster-Whisper backend":
        asr_backend = "faster_whisper"
    elif asr_backend_label == "Transformers backend":
        asr_backend = "transformers"
    else:
        asr_backend = "cascaded_encoder"

    input_mode = st.sidebar.radio("Input mode", ["Offline mode", "Streaming mode"])
    model_size = st.sidebar.selectbox("PhoWhisper model size", list(PHOWHISPER_MODELS), index=1)
    custom_model = st.sidebar.text_input("Custom PhoWhisper model/path", "")

    if asr_backend == "faster_whisper":
        default_model = default_ct2_model_path(model_size)
        help_text = "Use a local CTranslate2-converted PhoWhisper directory or a CT2 model repo."
    elif asr_backend == "cascaded_encoder":
        default_model = DEFAULT_CASCADE_ROOT
        help_text = (
            "Use a checkpoint exported by notebooks/train_cascaded_phowhisper_colab_kaggle.ipynb. "
            "The root may contain step_* folders; the newest step is selected automatically."
        )
    else:
        default_model = PHOWHISPER_MODELS.get(model_size) or DEFAULT_HF_MODEL
        help_text = "Use a Hugging Face PhoWhisper checkpoint, e.g. vinai/PhoWhisper-base."

    model_name_or_path = custom_model.strip() or default_model
    st.sidebar.caption(help_text)
    checkpoint_hint = local_checkpoint_hint(asr_backend, model_name_or_path)
    if checkpoint_hint:
        st.sidebar.warning(checkpoint_hint)

    device = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    compute_type = st.sidebar.selectbox(
        "Faster-Whisper compute type",
        ["auto", "float16", "int8_float16", "int8", "float32"],
        index=0,
        disabled=asr_backend != "faster_whisper",
    )
    beam_size = st.sidebar.slider("Beam size", min_value=1, max_value=8, value=5)

    with st.sidebar.expander("Streaming chunking"):
        chunk_seconds = st.slider("Chunk seconds", 2.0, 12.0, 6.0, 0.5)
        overlap_seconds = st.slider("Overlap seconds", 0.0, 2.0, 0.5, 0.1)
        min_rms = st.slider("Silence threshold", 0.001, 0.05, 0.008, 0.001)
        audio_receiver_size = st.select_slider(
            "WebRTC receiver queue",
            options=[256, 512, 1024, 2048, 4096],
            value=2048,
            help="Increase this if Streamlit reports a WebRTC queue overflow.",
        )
        if st.button("Reset stream"):
            st.session_state.stream_state = StreamingState(
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                min_rms=min_rms,
            )
            st.session_state.stream_result = {"segments": [], "vi_text": "", "en_text": ""}
            st.rerun()

    return {
        "asr_backend": asr_backend,
        "input_mode": input_mode,
        "model_name_or_path": model_name_or_path,
        "device": device,
        "compute_type": None if compute_type == "auto" else compute_type,
        "beam_size": beam_size,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "min_rms": min_rms,
        "audio_receiver_size": audio_receiver_size,
    }


def render_result(result: dict[str, object]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Vietnamese transcript")
        st.text_area("Vietnamese transcript", result.get("vi_text", ""), height=260, label_visibility="collapsed")
    with right:
        st.subheader("English translation")
        st.text_area("English translation", result.get("en_text", ""), height=260, label_visibility="collapsed")

    segments = result.get("segments", [])
    if segments:
        st.subheader("Segments")
        rows = [
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "vi_text": segment.vi_text,
                "en_text": segment.en_text,
            }
            for segment in segments
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def run_offline_ui(config: dict[str, object]) -> None:
    uploaded = st.file_uploader("Upload Vietnamese audio", type=["wav", "mp3", "m4a"])
    if not uploaded:
        st.info("Upload a wav, mp3, or m4a file to start offline transcription.")
        return

    st.audio(uploaded)
    if st.button("Transcribe and translate", type="primary"):
        try:
            suffix = Path(uploaded.name).suffix
            audio_path = save_uploaded_audio(uploaded, suffix)
            with st.status("Running PhoWhisper ASR and opus-mt-vi-en translation...", expanded=True):
                st.write("Loading or reusing cached models from Hugging Face/local cache.")
                result = run_offline(
                    audio=audio_path,
                    asr_backend=str(config["asr_backend"]),
                    model_name_or_path=str(config["model_name_or_path"]),
                    device=str(config["device"]),
                    compute_type=config["compute_type"],
                    beam_size=int(config["beam_size"]),
                )
                st.write("Finished.")
            render_result(result)
        except Exception as exc:
            st.error(f"Offline processing failed: {exc}")


def audio_frames_to_numpy(frames) -> tuple[int, np.ndarray] | None:
    if not frames:
        return None
    chunks = []
    sample_rate = frames[0].sample_rate
    for frame in frames:
        array = frame.to_ndarray()
        if array.ndim == 2:
            array = array.mean(axis=0)
        chunks.append(array.astype(np.float32) / 32768.0)
    return sample_rate, np.concatenate(chunks)


def run_streaming_ui(config: dict[str, object]) -> None:
    state: StreamingState = st.session_state.stream_state
    state.chunk_seconds = float(config["chunk_seconds"])
    state.overlap_seconds = float(config["overlap_seconds"])
    state.min_rms = float(config["min_rms"])

    st.caption("Allow microphone access in the browser. Finalized chunks appear below as the rolling buffer fills.")
    ctx = webrtc_streamer(
        key="phowhisper-mic",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=int(config["audio_receiver_size"]),
    )

    if not ctx.state.playing:
        st.warning("Microphone stream is stopped or permission has not been granted.")
        render_result(st.session_state.stream_result)
        return

    try:
        frames = ctx.audio_receiver.get_frames(timeout=1) if ctx.audio_receiver else []
    except Exception as exc:
        st.error(f"Could not read microphone frames: {exc}")
        return

    chunk = audio_frames_to_numpy(frames)
    if chunk is not None:
        try:
            st.session_state.stream_result = run_streaming(
                audio_chunk=chunk,
                asr_backend=str(config["asr_backend"]),
                state=state,
                model_name_or_path=str(config["model_name_or_path"]),
                device=str(config["device"]),
                compute_type=config["compute_type"],
                beam_size=min(int(config["beam_size"]), 3),
            )
        except Exception as exc:
            st.error(f"Streaming processing failed: {exc}")

    render_result(st.session_state.stream_result)
    st.rerun()


def main() -> None:
    init_state()
    config = sidebar_controls()

    st.title("faster-phowhisper")
    st.caption(
        "PhoWhisper Vietnamese ASR with Faster-Whisper/CTranslate2, Transformers, "
        "or a notebook-exported Cascaded Encoder, followed by explicit "
        "Helsinki-NLP/opus-mt-vi-en translation."
    )

    if config["input_mode"] == "Offline mode":
        run_offline_ui(config)
    else:
        run_streaming_ui(config)


if __name__ == "__main__":
    main()
