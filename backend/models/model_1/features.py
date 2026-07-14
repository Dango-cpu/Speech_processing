import base64
import tempfile
from pathlib import Path

try:
    import librosa
    import torch
    import torchaudio
except ImportError as exc:
    raise RuntimeError("Feature extraction requires librosa, torch, and torchaudio.") from exc


class MfccExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "center": True,
            },
        )

    def from_file(self, path):
        audio, _ = librosa.load(
            str(path),
            sr=self.sample_rate,
            mono=True,
        )
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        features = self.transform(waveform).squeeze(0).transpose(0, 1)
        return features

    def from_base64(self, payload, suffix=".webm"):
        audio_bytes = base64.b64decode(payload)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_bytes)
            temp_path = Path(audio_file.name)
        try:
            return self.from_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


def pad_feature_batch(items):
    lengths = torch.tensor([item.shape[0] for item in items], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
    return padded, lengths
