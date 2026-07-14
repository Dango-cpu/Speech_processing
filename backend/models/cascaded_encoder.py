import torch
import torch.nn as nn


class CascadedEncoder(nn.Module):
    """Experimental two-stage encoder interface.

    This is only the model-side contract. A useful cascaded Whisper/PhoWhisper
    model needs training or distillation so the decoder learns the refined
    encoder representation distribution.
    """

    def __init__(self, fast_encoder: nn.Module, refine_encoder: nn.Module):
        super().__init__()
        self.fast_encoder = fast_encoder
        self.refine_encoder = refine_encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        fast_features = self.fast_encoder(mel)
        refined_features = self.refine_encoder(fast_features)
        return refined_features


class CascadedWhisperASR:
    """Placeholder runtime for the experimental cascaded encoder path."""

    def __init__(self, config):
        self.config = config
        self.model = None

    def load(self):
        raise NotImplementedError(
            "The cascade backend is experimental and is not wired to a trained "
            "checkpoint yet. Train or distill a CascadedEncoder-compatible "
            "Whisper/PhoWhisper model, then integrate its encoder outputs with "
            "a decoder runtime."
        )

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError("Cascade backend is not inference-ready yet.")

    def unload(self):
        self.model = None
