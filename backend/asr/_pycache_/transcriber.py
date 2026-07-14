from backend.asr.config import ASRConfig, ASRBackend, ComputeType


class ASRTranscriber:
    """Unified ASR entry point for torch, faster-whisper, and experiments."""

    def __init__(
        self,
        model_path: str | None = None,
        backend: ASRBackend = "torch",
        device: str | None = None,
        compute_type: ComputeType = "float16",
        language: str = "vi",
        beam_size: int = 3,
        max_new_tokens: int = 96,
    ):
        self.config = ASRConfig(
            model_path=model_path,
            backend=backend,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
        )
        self._impl = None

    def load(self):
        if self._impl is not None:
            return self

        if self.config.backend == "torch":
            from backend.models.model3_whisper import WhisperASR

            self._impl = WhisperASR(
                model_name=self.config.resolved_model_path,
                device=self.config.device,
                language=self.config.language,
                beam_size=self.config.beam_size,
                max_new_tokens=self.config.max_new_tokens,
            )
        elif self.config.backend == "faster_whisper":
            from backend.models.faster_whisper_asr import FasterWhisperASR

            self._impl = FasterWhisperASR(
                model_path=self.config.resolved_model_path,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language,
                beam_size=self.config.beam_size,
            )
        elif self.config.backend == "cascade":
            from backend.models.cascaded_encoder import CascadedWhisperASR

            self._impl = CascadedWhisperASR(self.config)
        else:
            raise ValueError(f"Unsupported ASR backend: {self.config.backend}")

        self._impl.load()
        return self

    def transcribe(self, audio_path: str) -> str:
        if self._impl is None:
            self.load()
        return self._impl.transcribe(audio_path)

    def unload(self):
        if self._impl is not None:
            self._impl.unload()
        self._impl = None
