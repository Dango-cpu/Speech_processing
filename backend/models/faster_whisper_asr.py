from pathlib import Path

import torch

from backend.utils.device import get_device


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CTRANSLATE2_MODEL_DIR = ROOT_DIR / "checkpoints" / "phowhisper-base-ct2"
DEFAULT_HF_MODEL = "vinai/PhoWhisper-base"


class FasterWhisperASR:
    """CTranslate2/faster-whisper inference backend.

    `model_path` may be a converted CTranslate2 directory or a faster-whisper
    compatible model id. For local production use, prefer a converted directory.
    """

    def __init__(
        self,
        model_path=None,
        device=None,
        compute_type="float16",
        language="vi",
        beam_size=3,
    ):
        self.model_path = str(model_path or self._default_model_path())
        self.uses_default_model_path = model_path is None
        self.device = device or get_device()
        self.compute_type = self._safe_compute_type(compute_type)
        self.language = language
        self.beam_size = beam_size
        self.model = None

    def _default_model_path(self):
        if DEFAULT_CTRANSLATE2_MODEL_DIR.exists():
            return DEFAULT_CTRANSLATE2_MODEL_DIR
        return DEFAULT_HF_MODEL

    def _safe_compute_type(self, compute_type):
        if self.device == "cpu" and compute_type in {"float16", "int8_float16"}:
            return "int8"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            if compute_type in {"float16", "int8_float16"}:
                return "int8"
        return compute_type

    def load(self):
        if self.model is not None:
            return

        if (
            self.uses_default_model_path
            and self.model_path == DEFAULT_HF_MODEL
            and not DEFAULT_CTRANSLATE2_MODEL_DIR.exists()
        ):
            raise FileNotFoundError(
                "The default faster PhoWhisper backend requires a converted "
                f"CTranslate2 model at {DEFAULT_CTRANSLATE2_MODEL_DIR}. Run "
                "`python scripts/convert_whisper_to_ctranslate2.py --model "
                "vinai/PhoWhisper-base --output_dir "
                "checkpoints/phowhisper-base-ct2 --quantization float16`, "
                "or pass model_path to an existing converted PhoWhisper model."
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. Install optional ASR "
                "dependencies with `uv pip install faster-whisper ctranslate2` "
                "or add them to your environment before using this backend."
            ) from exc

        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

        print("FasterWhisper ready.")
        print("Model:", self.model_path)
        print("Device:", self.device)
        print("Compute type:", self.compute_type)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load()

        segments, _ = self.model.transcribe(
            audio_path,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=False,
        )

        return " ".join(segment.text.strip() for segment in segments).strip()

    def unload(self):
        if self.model is not None:
            del self.model
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
