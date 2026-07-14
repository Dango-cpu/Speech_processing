import os
from pathlib import Path

import torch
import librosa
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForCTC

from backend.utils.device import get_device


ROOT_DIR = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = ROOT_DIR / "checkpoints" / "wav2vec2-base-vietnamese"

load_dotenv(ROOT_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


class Wav2Vec2ASR:
    def __init__(self, model_name="dragonSwing/wav2vec2-base-vietnamese"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = get_device()

    def _load_from_local(self):
        self.processor = AutoProcessor.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
        )

        self.model = AutoModelForCTC.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
            use_safetensors=True,
        )

    def _download_and_save_once(self):
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        hf_token = os.getenv("HF_TOKEN")

        print("Local Wav2Vec2 not found. Downloading once...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=hf_token,
        )

        self.model = AutoModelForCTC.from_pretrained(
            self.model_name,
            token=hf_token,
            use_safetensors=True,
        )

        self.processor.save_pretrained(LOCAL_MODEL_DIR)
        self.model.save_pretrained(
            LOCAL_MODEL_DIR,
            safe_serialization=True,
        )

        print(f"Saved Wav2Vec2 to: {LOCAL_MODEL_DIR}")

    def load(self):
        if self.model is not None:
            return

        try:
            print(f"Trying to load Wav2Vec2 from local: {LOCAL_MODEL_DIR}")
            self._load_from_local()
            print("Loaded Wav2Vec2 from local folder.")
        except Exception as e:
            print("Cannot load local Wav2Vec2.")
            print("Reason:", e)

            self._download_and_save_once()

        self.model.to(self.device)
        self.model.eval()

        print("Wav2Vec2 ready.")
        print("Device:", self.device)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load()

        audio, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
        )

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        text = self.processor.batch_decode(predicted_ids)[0]

        return text.strip()

    def unload(self):
        if self.model is not None:
            del self.model

        if self.processor is not None:
            del self.processor

        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
