import os
from pathlib import Path

import torch
import librosa
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from backend.utils.device import get_device, get_torch_dtype


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "vinai/PhoWhisper-base"
LOCAL_MODEL_DIR = ROOT_DIR / "checkpoints" / "phowhisper-base"

load_dotenv(ROOT_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


class WhisperASR:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        device=None,
        language="vi",
        beam_size=3,
        max_new_tokens=96,
    ):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.processor = None
        self.model = None
        self.device = device or get_device()
        self.torch_dtype = get_torch_dtype()
        self.language = language
        self.beam_size = beam_size
        self.max_new_tokens = max_new_tokens

    def _local_model_dir(self):
        model_path = Path(str(self.model_name))
        if model_path.exists():
            return model_path
        if self.model_name == DEFAULT_MODEL_NAME:
            return LOCAL_MODEL_DIR
        return None

    def _load_from_local(self):
        local_model_dir = self._local_model_dir()
        if local_model_dir is None:
            raise FileNotFoundError(
                f"No local model directory configured for {self.model_name}"
            )

        self.processor = WhisperProcessor.from_pretrained(
            local_model_dir,
            language=self.language,
            task="transcribe",
            local_files_only=True,
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            local_model_dir,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )

    def _download_and_save_once(self):
        local_model_dir = self._local_model_dir() or LOCAL_MODEL_DIR
        local_model_dir.mkdir(parents=True, exist_ok=True)
        hf_token = os.getenv("HF_TOKEN")

        print("Local model not found. Downloading PhoWhisper once...")

        processor = WhisperProcessor.from_pretrained(
            self.model_name,
            language=self.language,
            task="transcribe",
            token=hf_token,
        )

        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )

        processor.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)

        print(f"Saved PhoWhisper to: {local_model_dir}")

        del processor
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self):
        if self.model is not None:
            return

        try:
            print(f"Trying to load PhoWhisper from local folder: {self._local_model_dir()}")
            self._load_from_local()
            print("Loaded PhoWhisper from local folder.")
        except Exception as e:
            print("Cannot load local PhoWhisper.")
            print("Reason:", e)

            self._download_and_save_once()

            print("Loading PhoWhisper again from local folder...")
            self._load_from_local()

        self.model.to(self.device)
        self.model.eval()

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task="transcribe",
        )
        self.model.config.forced_decoder_ids = forced_decoder_ids

        print("PhoWhisper ready.")
        print("Device:", self.device)
        print("Dtype:", self.torch_dtype)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load()

        audio, _ = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.torch_dtype,
        )

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.beam_size,
                no_repeat_ngram_size=3,
            )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

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
