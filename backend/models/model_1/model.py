try:
    import torch
    from torch import nn
except ImportError as exc:
    raise RuntimeError("Model 1 requires torch. Install it with: pip install torch torchaudio") from exc


class CnnBiLstmCtc(nn.Module):
    """CNN + Bi-LSTM acoustic model trained with CTC loss.

    Input shape: (batch, time, n_mfcc)
    Output shape: (time, batch, vocab_size), suitable for torch.nn.CTCLoss.
    """

    def __init__(
        self,
        n_mfcc=40,
        vocab_size=30,
        cnn_channels=64,
        lstm_hidden=256,
        lstm_layers=3,
        dropout=0.15,
    ):
        super().__init__()
        self.feature_norm = nn.LayerNorm(n_mfcc)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )
        reduced_features = (n_mfcc // 4) * cnn_channels
        self.recurrent = nn.LSTM(
            input_size=reduced_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, vocab_size),
        )

    def forward(self, features, feature_lengths=None):
        features = self.feature_norm(features)
        x = features.unsqueeze(1)
        x = self.cnn(x)
        batch, channels, time_steps, feature_bins = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, time_steps, channels * feature_bins)
        x, _ = self.recurrent(x)
        logits = self.classifier(x)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        return log_probs, feature_lengths


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = CnnBiLstmCtc(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
