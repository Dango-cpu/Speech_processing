from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ASRBackend = Literal["torch", "faster_whisper", "cascade"]
ComputeType = Literal["float32", "float16", "int8_float16", "int8"]


@dataclass(frozen=True)
class ASRConfig:
    model_path: str | Path | None = None
    backend: ASRBackend = "torch"
    device: str | None = None
    compute_type: ComputeType = "float16"
    language: str = "vi"
    beam_size: int = 3
    max_new_tokens: int = 96

    @property
    def resolved_model_path(self) -> str | None:
        if self.model_path is None:
            return None
        return str(self.model_path)
