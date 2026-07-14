import argparse
import subprocess
import sys
from pathlib import Path


VALID_QUANTIZATION = {
    "float32",
    "float16",
    "bfloat16",
    "int16",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_bfloat16",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Whisper/PhoWhisper Hugging Face checkpoint to CTranslate2."
    )
    parser.add_argument(
        "--model",
        default="vinai/PhoWhisper-base",
        help=(
            "Hugging Face model id or local Hugging Face-format model directory. "
            "Defaults to vinai/PhoWhisper-base."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/phowhisper-base-ct2",
        help="Directory where the converted CTranslate2 model will be written.",
    )
    parser.add_argument(
        "--quantization",
        default="float16",
        choices=sorted(VALID_QUANTIZATION),
        help="CTranslate2 weight quantization. Use int8 for CPU-friendly models.",
    )
    parser.add_argument(
        "--copy_files",
        nargs="*",
        default=["tokenizer.json", "preprocessor_config.json"],
        help="Tokenizer/processor files to copy into the CTranslate2 directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. "
            "Pass --force to reuse it."
        )

    command = [
        sys.executable,
        "-m",
        "ctranslate2.converters.transformers",
        "--model",
        args.model,
        "--output_dir",
        str(output_dir),
        "--quantization",
        args.quantization,
    ]

    if args.copy_files:
        command.append("--copy_files")
        command.extend(args.copy_files)

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    print(f"Converted model written to: {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
