from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
TARGET_SAMPLE_RATE = 16_000


def save_uploaded_audio(uploaded_file: BinaryIO, suffix: str) -> str:
    suffix = suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. Upload wav, mp3, or m4a audio."
        )

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_audio_mono(audio: str | tuple[int, object], sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if isinstance(audio, tuple):
        source_sr, samples = audio
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=1)
        if source_sr != sr:
            array = librosa.resample(array, orig_sr=source_sr, target_sr=sr)
        return normalize_audio(array)

    path = Path(audio)
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. Use wav, mp3, or m4a."
        )

    array, _ = librosa.load(path, sr=sr, mono=True)
    return normalize_audio(array)


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak
    return samples


def write_temp_wav(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, samples, sample_rate)
        return tmp.name


def has_speech(
    samples: np.ndarray,
    rms_threshold: float = 0.008,
    min_duration_seconds: float = 0.75,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> bool:
    if samples.size < int(min_duration_seconds * sample_rate):
        return False
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return rms >= rms_threshold
