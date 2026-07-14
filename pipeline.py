from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asr import ASRSegment, transcribe
from asr.audio import TARGET_SAMPLE_RATE, has_speech


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    vi_text: str


@dataclass
class StreamingState:
    sample_rate: int = TARGET_SAMPLE_RATE
    chunk_seconds: float = 6.0
    overlap_seconds: float = 0.5
    min_rms: float = 0.008
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    elapsed_seconds: float = 0.0
    finalized: list[TranscriptSegment] = field(default_factory=list)


def normalize_segments(segments: list[ASRSegment]) -> list[TranscriptSegment]:
    normalized: list[TranscriptSegment] = []
    for segment in segments:
        vi_text = segment.text.strip()
        if not vi_text:
            continue
        normalized.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                vi_text=vi_text,
            )
        )
    return normalized


def run_offline(
    audio: str,
    asr_backend: str,
    model_name_or_path: str,
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 5,
) -> dict[str, object]:
    segments = transcribe(
        audio=audio,
        backend=asr_backend,
        model_name_or_path=model_name_or_path,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=asr_backend == "faster_whisper",
    )
    return build_result(normalize_segments(segments))


def run_streaming(
    audio_chunk: tuple[int, np.ndarray],
    asr_backend: str,
    state: StreamingState,
    model_name_or_path: str,
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 3,
) -> dict[str, object]:
    sample_rate, samples = audio_chunk
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    if sample_rate != state.sample_rate:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=state.sample_rate)

    state.buffer = np.concatenate([state.buffer, samples])
    chunk_size = int(state.chunk_seconds * state.sample_rate)
    if state.buffer.size < chunk_size:
        return build_result(state.finalized)

    current = state.buffer[:chunk_size]
    overlap = int(state.overlap_seconds * state.sample_rate)
    state.buffer = state.buffer[max(0, chunk_size - overlap) :]

    start_offset = state.elapsed_seconds
    state.elapsed_seconds += (chunk_size - overlap) / state.sample_rate

    if not has_speech(
        current,
        rms_threshold=state.min_rms,
        sample_rate=state.sample_rate,
    ):
        return build_result(state.finalized)

    segments = transcribe(
        audio=(state.sample_rate, current),
        backend=asr_backend,
        model_name_or_path=model_name_or_path,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=asr_backend == "faster_whisper",
    )
    for segment in normalize_segments(segments):
        state.finalized.append(
            TranscriptSegment(
                start=segment.start + start_offset,
                end=segment.end + start_offset,
                vi_text=segment.vi_text,
            )
        )

    return build_result(state.finalized)


def build_result(segments: list[TranscriptSegment]) -> dict[str, object]:
    return {
        "segments": segments,
        "vi_text": " ".join(segment.vi_text for segment in segments).strip(),
    }
