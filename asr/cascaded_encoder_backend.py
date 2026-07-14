from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .audio import TARGET_SAMPLE_RATE, load_audio_mono
from .common import ASRSegment
from .transformers_backend import resolve_device


CHECKPOINT_FILE = "cascaded_phowhisper.pt"


@dataclass(slots=True)
class CascadedConfig:
    model_id: str
    refine_start_layer: int
    fast_hidden_size: int
    fast_num_layers: int
    fast_num_heads: int
    fast_ffn_dim: int
    language: str = "vi"
    task: str = "transcribe"
    max_new_tokens: int = 96


class LightweightFastEncoderStage(nn.Module):
    """Runtime copy of the notebook fast encoder stage."""

    def __init__(
        self,
        config,
        fast_hidden_size: int,
        fast_num_layers: int,
        fast_num_heads: int,
        fast_ffn_dim: int,
    ):
        super().__init__()
        num_mel_bins = getattr(config, "num_mel_bins", 80)
        self.conv1 = nn.Conv1d(num_mel_bins, fast_hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            fast_hidden_size,
            fast_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, fast_hidden_size)
        self.dropout = config.dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fast_hidden_size,
            nhead=fast_num_heads,
            dim_feedforward=fast_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=fast_num_layers)
        self.fast_norm = nn.LayerNorm(fast_hidden_size)
        self.to_phowhisper_width = nn.Linear(fast_hidden_size, config.d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        positions = self.embed_positions.weight[: hidden_states.size(1)]
        hidden_states = hidden_states + positions.unsqueeze(0).to(hidden_states.dtype)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.layers(hidden_states)
        hidden_states = self.fast_norm(hidden_states)
        return self.to_phowhisper_width(hidden_states)


class WhisperRefineEncoderStage(nn.Module):
    def __init__(self, whisper_encoder: nn.Module, refine_start_layer: int):
        super().__init__()
        self.embed_positions = whisper_encoder.embed_positions
        self.layers = nn.ModuleList(list(whisper_encoder.layers[refine_start_layer:]))
        self.layer_norm = whisper_encoder.layer_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        positions = self.embed_positions.weight[: hidden_states.size(1)]
        hidden_states = hidden_states + positions.unsqueeze(0).to(hidden_states.dtype)
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=None)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        return self.layer_norm(hidden_states)


class CascadedEncoder(nn.Module):
    def __init__(self, fast_encoder: nn.Module, refine_encoder: nn.Module):
        super().__init__()
        self.fast_encoder = fast_encoder
        self.refine_encoder = refine_encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.refine_encoder(self.fast_encoder(input_features))


class CascadedPhoWhisperForConditionalGeneration(nn.Module):
    """Inference wrapper matching the Colab/Kaggle training notebook export."""

    def __init__(
        self,
        base_model: WhisperForConditionalGeneration,
        cfg: CascadedConfig,
    ):
        super().__init__()
        self.config = base_model.config
        encoder = base_model.model.encoder
        self.encoder = CascadedEncoder(
            LightweightFastEncoderStage(
                self.config,
                cfg.fast_hidden_size,
                cfg.fast_num_layers,
                cfg.fast_num_heads,
                cfg.fast_ffn_dim,
            ),
            WhisperRefineEncoderStage(encoder, cfg.refine_start_layer),
        )
        self.decoder = base_model.model.decoder
        self.proj_out = base_model.proj_out

    def forward(self, input_features, decoder_input_ids):
        encoder_hidden_states = self.encoder(input_features)
        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return self.proj_out(decoder_outputs.last_hidden_state)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_features: torch.Tensor,
        processor: WhisperProcessor,
        language: str = "vi",
        task: str = "transcribe",
        max_new_tokens: int = 96,
    ) -> torch.Tensor:
        self.eval()
        prompt = [self.config.decoder_start_token_id]
        for _position, token_id in processor.get_decoder_prompt_ids(
            language=language,
            task=task,
        ):
            prompt.append(token_id)

        decoder_input_ids = torch.tensor([prompt], device=input_features.device)
        for _ in range(max_new_tokens):
            logits = self(input_features, decoder_input_ids=decoder_input_ids)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id:
                break
        return decoder_input_ids


def resolve_checkpoint_file(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.is_file():
        return path
    if (path / CHECKPOINT_FILE).exists():
        return path / CHECKPOINT_FILE

    step_dirs = [
        item
        for item in path.glob("step_*")
        if item.is_dir() and (item / CHECKPOINT_FILE).exists()
    ]
    if step_dirs:
        return sorted(step_dirs, key=_step_number)[-1] / CHECKPOINT_FILE

    raise FileNotFoundError(
        "Could not find a cascaded encoder checkpoint. Point this backend at "
        f"a notebook export containing {CHECKPOINT_FILE}, for example "
        "checkpoints/cascaded_phowhisper_ckpt/step_500."
    )


def _step_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def checkpoint_to_config(checkpoint: dict) -> CascadedConfig:
    required = [
        "model_id",
        "refine_start_layer",
        "fast_hidden_size",
        "fast_num_layers",
        "fast_num_heads",
        "fast_ffn_dim",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            "The cascaded checkpoint is missing notebook export fields: "
            + ", ".join(missing)
        )

    return CascadedConfig(
        model_id=str(checkpoint["model_id"]),
        refine_start_layer=int(checkpoint["refine_start_layer"]),
        fast_hidden_size=int(checkpoint["fast_hidden_size"]),
        fast_num_layers=int(checkpoint["fast_num_layers"]),
        fast_num_heads=int(checkpoint["fast_num_heads"]),
        fast_ffn_dim=int(checkpoint["fast_ffn_dim"]),
    )


@st.cache_resource(show_spinner="Loading notebook-exported Cascaded PhoWhisper...")
def load_cascaded_encoder(checkpoint_path: str, device: str):
    checkpoint_file = resolve_checkpoint_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    cfg = checkpoint_to_config(checkpoint)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("The cascaded checkpoint is missing model_state_dict.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model_source = os.getenv("CASCADED_BASE_MODEL_PATH", cfg.model_id)
    processor_dir = checkpoint_file.parent / "processor"
    processor_source = str(processor_dir) if processor_dir.exists() else cfg.model_id
    processor = WhisperProcessor.from_pretrained(processor_source)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_source,
        torch_dtype=dtype,
    )
    model = CascadedPhoWhisperForConditionalGeneration(base_model, cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return processor, model, cfg, dtype


def transcribe_cascaded_encoder(
    audio: str | tuple[int, object],
    checkpoint_path: str,
    device: str = "auto",
    beam_size: int = 1,
) -> list[ASRSegment]:
    del beam_size
    resolved_device = resolve_device(device)
    processor, model, cfg, dtype = load_cascaded_encoder(checkpoint_path, resolved_device)
    samples = load_audio_mono(audio)
    duration = samples.size / TARGET_SAMPLE_RATE
    inputs = processor(
        samples,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=resolved_device, dtype=dtype)

    generated_ids = model.generate_greedy(
        input_features,
        processor=processor,
        language=cfg.language,
        task=cfg.task,
        max_new_tokens=cfg.max_new_tokens,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return [ASRSegment(start=0.0, end=float(duration), text=text)] if text else []
