import mimetypes
from pathlib import Path

import torch

from backend.models.model_1.features import MfccExtractor, pad_feature_batch
from backend.models.model_1.metrics import Timer, real_time_factor
from backend.models.model_1.model import load_model
from backend.models.model_1.text import TextTransform


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "model1.pt"


def suffix_for_mime(mime_type):
    if not mime_type:
        return ".webm"
    return mimetypes.guess_extension(mime_type.split(";")[0]) or ".webm"


class Model1ASR:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.checkpoint = None
        self.text = None
        self.extractor = None

    def load(self):
        if self.model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {self.checkpoint_path}")

        self.model, self.checkpoint = load_model(
            str(self.checkpoint_path),
            device=self.device,
        )

        self.model.eval()

        self.text = TextTransform(self.checkpoint.get("vocab"))

        self.extractor = MfccExtractor(
            sample_rate=self.checkpoint.get("sample_rate", 16000),
            n_mfcc=self.checkpoint.get("config", {}).get("n_mfcc", 40),
        )

        print("Model 1 loaded.")
        print("Checkpoint:", self.checkpoint_path)
        print("Device:", self.device)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load()

        features = self.extractor.from_file(audio_path)
        batch, lengths = pad_feature_batch([features])

        with Timer() as timer, torch.no_grad():
            log_probs, _ = self.model(
                batch.to(self.device),
                lengths.to(self.device),
            )
            prediction = log_probs.argmax(dim=-1)[:, 0].detach().cpu().tolist()

        transcript = self.text.decode_greedy(prediction)

        print("Model 1 processing seconds:", timer.elapsed_seconds)
        print("Model 1 RTF:", real_time_factor(timer.elapsed_seconds, None))

        return transcript

    def unload(self):
        if self.model is not None:
            del self.model

        self.model = None
        self.checkpoint = None
        self.text = None
        self.extractor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe(payload):
    checkpoint_path = payload.get("checkpoint") or DEFAULT_CHECKPOINT
    asr = Model1ASR(checkpoint_path=checkpoint_path)
    asr.load()

    if payload.get("audio_path"):
        transcript = asr.transcribe(payload["audio_path"])
    else:
        raise ValueError("audio_path is required")

    return {
        "provider": "model_1",
        "isMock": False,
        "transcript": transcript,
    }