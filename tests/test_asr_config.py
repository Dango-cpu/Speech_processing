import unittest

from backend.asr import ASRConfig, ASRTranscriber
from backend.models.cascaded_encoder import CascadedEncoder


class ASRConfigTest(unittest.TestCase):
    def test_default_config_is_torch_vietnamese(self):
        config = ASRConfig()

        self.assertEqual(config.backend, "torch")
        self.assertEqual(config.language, "vi")
        self.assertIsNone(config.resolved_model_path)

    def test_transcriber_stores_backend_options(self):
        transcriber = ASRTranscriber(
            model_path="checkpoints/phowhisper-base-ct2",
            backend="faster_whisper",
            device="cpu",
            compute_type="int8",
            language="vi",
        )

        self.assertEqual(transcriber.config.backend, "faster_whisper")
        self.assertEqual(transcriber.config.compute_type, "int8")
        self.assertEqual(
            transcriber.config.resolved_model_path,
            "checkpoints/phowhisper-base-ct2",
        )

    def test_cascaded_encoder_is_torch_module(self):
        import torch.nn as nn

        encoder = CascadedEncoder(nn.Identity(), nn.Identity())

        self.assertIsInstance(encoder, nn.Module)


if __name__ == "__main__":
    unittest.main()
