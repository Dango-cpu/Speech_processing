from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from faster_whisper import WhisperModel

from .audio import write_temp_wav
from .common import ASRSegment


def is_probably_local_path(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path)
    normalized = model_name_or_path.replace("\\", "/")
    return (
        path.exists()
        or path.is_absolute()
        or "\\" in model_name_or_path
        or normalized.startswith(("./", "../", "checkpoints/"))
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA was selected but is unavailable. Falling back to CPU.")
        return "cpu"
    return device


def default_compute_type(device: str, compute_type: str | None = None) -> str:
    if compute_type and compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


@st.cache_resource(show_spinner="Loading CTranslate2 PhoWhisper model...")
def load_faster_whisper_model(
    model_name_or_path: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    if not model_name_or_path:
        raise ValueError("Choose a CTranslate2-converted PhoWhisper checkpoint.")

    if is_probably_local_path(model_name_or_path):
        path = Path(model_name_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"CTranslate2 checkpoint not found: {model_name_or_path}"
            )
        if not (path / "model.bin").exists():
            raise FileNotFoundError(
                f"{model_name_or_path} is missing model.bin. Convert PhoWhisper to "
                "CTranslate2 before using the faster-whisper backend."
            )

    return WhisperModel(model_name_or_path, device=device, compute_type=compute_type)


def transcribe_faster_whisper(
    audio: str | tuple[int, object],
    model_name_or_path: str,
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> list[ASRSegment]:
    resolved_device = resolve_device(device)
    resolved_compute_type = default_compute_type(resolved_device, compute_type)
    model = load_faster_whisper_model(
        model_name_or_path=model_name_or_path,
        device=resolved_device,
        compute_type=resolved_compute_type,
    )

    audio_input = audio
    if isinstance(audio, tuple):
        from .audio import load_audio_mono

        audio_input = write_temp_wav(load_audio_mono(audio))

    segments, _info = model.transcribe(
        audio_input,
        language="vi",
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
    )

    return [
        ASRSegment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
