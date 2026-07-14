from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ASRBackend = Literal["faster_whisper", "transformers", "cascaded_encoder"]


@dataclass(slots=True)
class ASRSegment:
    start: float
    end: float
    text: str


def transcribe(
    audio: str | tuple[int, object],
    backend: ASRBackend,
    model_name_or_path: str,
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> list[ASRSegment]:
    if backend == "faster_whisper":
        from .faster_whisper_backend import transcribe_faster_whisper

        return transcribe_faster_whisper(
            audio=audio,
            model_name_or_path=model_name_or_path,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

    if backend == "transformers":
        from .transformers_backend import transcribe_transformers

        return transcribe_transformers(
            audio=audio,
            model_name_or_path=model_name_or_path,
            device=device,
            beam_size=beam_size,
        )

    if backend == "cascaded_encoder":
        from .cascaded_encoder_backend import transcribe_cascaded_encoder

        return transcribe_cascaded_encoder(
            audio=audio,
            checkpoint_path=model_name_or_path,
            device=device,
            beam_size=beam_size,
        )

    raise ValueError(f"Unsupported ASR backend: {backend}")
