from __future__ import annotations

import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .audio import TARGET_SAMPLE_RATE, load_audio_mono
from .common import ASRSegment


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA was selected but is unavailable. Falling back to CPU.")
        return "cpu"
    return device


@st.cache_resource(show_spinner="Loading Transformers PhoWhisper model...")
def load_transformers_model(model_name_or_path: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return processor, model, dtype


def transcribe_transformers(
    audio: str | tuple[int, object],
    model_name_or_path: str,
    device: str = "auto",
    beam_size: int = 5,
) -> list[ASRSegment]:
    resolved_device = resolve_device(device)
    processor, model, dtype = load_transformers_model(model_name_or_path, resolved_device)
    samples = load_audio_mono(audio)
    duration = samples.size / TARGET_SAMPLE_RATE

    inputs = processor(
        samples,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=resolved_device, dtype=dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi",
        task="transcribe",
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=beam_size,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return [ASRSegment(start=0.0, end=float(duration), text=text)] if text else []
