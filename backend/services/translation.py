import os
import re
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = ROOT_DIR / "checkpoints" / "opus-mt-vi-en"
HF_CACHE_DIR = ROOT_DIR / "checkpoints" / "hf_cache"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

LANGUAGES = {
    "English": "en",
}


class TranslationService:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-vi-en"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.torch_dtype = None
        self.torch = None
        self.auto_model = None
        self.auto_tokenizer = None
        self.snapshot_download = None

    def local_model_exists(self):
        if not LOCAL_MODEL_DIR.exists():
            return False

        has_config = (LOCAL_MODEL_DIR / "config.json").exists()
        has_tokenizer = any(
            (LOCAL_MODEL_DIR / filename).exists()
            for filename in (
                "tokenizer.json",
                "sentencepiece.bpe.model",
                "spm.model",
                "source.spm",
                "target.spm",
                "vocab.json",
            )
        )
        weight_patterns = ("*.safetensors", "pytorch_model*.bin")
        has_weights = any(
            weight_file.exists()
            for pattern in weight_patterns
            for weight_file in LOCAL_MODEL_DIR.glob(pattern)
        )
        return has_config and has_tokenizer and has_weights

    def _ensure_dependencies(self):
        if self.torch is not None:
            return

        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Translation requires torch, transformers, and huggingface-hub. "
                "Install the project dependencies with `uv sync` before using translation."
            ) from exc

        from backend.utils.device import get_device, get_torch_dtype

        self.torch = torch
        self.auto_model = AutoModelForSeq2SeqLM
        self.auto_tokenizer = AutoTokenizer
        self.snapshot_download = snapshot_download
        self.device = get_device()
        self.torch_dtype = get_torch_dtype()

    def _load_from_local(self):
        self.tokenizer = self.auto_tokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
        )
        self.model = self.auto_model.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )

    def _model_cache_dir(self):
        cache_name = f"models--{self.model_name.replace('/', '--')}"
        return HF_CACHE_DIR / cache_name

    def _remove_project_dir(self, target_dir):
        target_dir = target_dir.resolve()
        checkpoints_dir = (ROOT_DIR / "checkpoints").resolve()

        if target_dir.exists() and checkpoints_dir in target_dir.parents:
            shutil.rmtree(target_dir)

    def _clear_download_artifacts(self):
        self._remove_project_dir(LOCAL_MODEL_DIR)
        self._remove_project_dir(self._model_cache_dir())

    def _download_snapshot_to_local_dir(self):
        hf_token = os.getenv("HF_TOKEN")

        self.snapshot_download(
            repo_id=self.model_name,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

    def _download_pretrained(self):
        self._download_snapshot_to_local_dir()

        tokenizer = self.auto_tokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
        )
        model = self.auto_model.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )
        return tokenizer, model

    def _download_pretrained_via_cache(self, force_download=True):
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hf_token = os.getenv("HF_TOKEN")

        tokenizer = self.auto_tokenizer.from_pretrained(
            self.model_name,
            cache_dir=HF_CACHE_DIR,
            force_download=force_download,
            token=hf_token,
        )
        model = self.auto_model.from_pretrained(
            self.model_name,
            cache_dir=HF_CACHE_DIR,
            force_download=force_download,
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )
        return tokenizer, model

    def _download_and_save_once(self):
        self._clear_download_artifacts()
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self.tokenizer, self.model = self._download_pretrained()
        except Exception as exc:
            self._clear_download_artifacts()
            try:
                self.tokenizer, self.model = self._download_pretrained_via_cache()
            except Exception as fallback_exc:
                self._clear_download_artifacts()
                raise RuntimeError(
                    "The Vietnamese-English translator model could not be downloaded. "
                    f"Original error: {exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc

        self.tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        self.model.save_pretrained(LOCAL_MODEL_DIR, safe_serialization=True)

    def load(self, progress_callback=None):
        if self.model is not None:
            return

        self._ensure_dependencies()

        if self.local_model_exists():
            if progress_callback:
                progress_callback("Loading translation model from local checkpoint...")
            self._load_from_local()
        else:
            if progress_callback:
                progress_callback(
                    "No local translation model found. Downloading Vietnamese-English translator..."
                )
            self._download_and_save_once()

        self.model.to(self.device)
        self.model.eval()

        if progress_callback:
            progress_callback("Translation model ready.")

    def translate(self, text, target_lang="eng_Latn", source_lang="vie_Latn"):
        if not text or not text.strip():
            return ""

        if self.model is None:
            self.load()

        forced_bos_token_id = None
        if self.model_name.startswith("facebook/nllb"):
            self.tokenizer.src_lang = source_lang
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        translated_chunks = [
            self._translate_chunk(chunk.strip(), forced_bos_token_id)
            for chunk in self._chunk_text(text)
            if chunk.strip()
        ]
        return " ".join(translated_chunks).strip()

    def _translate_chunk(self, text, forced_bos_token_id=None):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with self.torch.no_grad():
            generation_kwargs = {
                "max_length": 512,
                "num_beams": 2,
            }
            if forced_bos_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

            generated_tokens = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

    def _chunk_text(self, text, max_chars=900):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current = ""

        for sentence in sentences:
            if not sentence:
                continue

            next_chunk = f"{current} {sentence}".strip()
            if len(next_chunk) <= max_chars:
                current = next_chunk
                continue

            if current:
                chunks.append(current)

            if len(sentence) <= max_chars:
                current = sentence
            else:
                chunks.extend(
                    sentence[index:index + max_chars]
                    for index in range(0, len(sentence), max_chars)
                )
                current = ""

        if current:
            chunks.append(current)

        return chunks

    def unload(self):
        if self.model is not None:
            del self.model

        if self.tokenizer is not None:
            del self.tokenizer

        self.model = None
        self.tokenizer = None

        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
