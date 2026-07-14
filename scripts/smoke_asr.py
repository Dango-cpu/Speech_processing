import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backend.asr import ASRTranscriber


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test ASR backends.")
    parser.add_argument("--audio", required=True, help="Path to a short Vietnamese audio file.")
    parser.add_argument("--torch_model", default=None, help="HF model id or local torch model dir.")
    parser.add_argument(
        "--ct2_model",
        default=None,
        help="Converted CTranslate2 model dir or faster-whisper model id.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--compute_type",
        default="int8",
        choices=["float32", "float16", "int8_float16", "int8"],
    )
    parser.add_argument("--language", default="vi")
    return parser.parse_args()


def run_backend(name, transcriber):
    transcriber.load()
    text = transcriber.transcribe(args.audio)
    transcriber.unload()
    print(f"[{name}] {text}")
    return text


if __name__ == "__main__":
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    outputs = {}

    if args.torch_model:
        outputs["torch"] = run_backend(
            "torch",
            ASRTranscriber(
                model_path=args.torch_model,
                backend="torch",
                device=args.device,
                language=args.language,
            ),
        )

    if args.ct2_model:
        outputs["faster_whisper"] = run_backend(
            "faster_whisper",
            ASRTranscriber(
                model_path=args.ct2_model,
                backend="faster_whisper",
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
            ),
        )

    if not outputs:
        raise SystemExit("Pass --torch_model, --ct2_model, or both.")

    if len(outputs) == 2:
        print("[compare] Outputs are identical:", outputs["torch"] == outputs["faster_whisper"])
