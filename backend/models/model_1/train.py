import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.models.model_1.features import MfccExtractor, pad_feature_batch
from backend.models.model_1.metrics import char_error_rate, word_error_rate
from backend.models.model_1.model import CnnBiLstmCtc
from backend.models.model_1.text import TextTransform


class SpeechCsvDataset(Dataset):
    def __init__(self, csv_path, extractor, text):
        self.rows = []
        self.extractor = extractor
        self.text = text
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("audio_path") and row.get("transcript"):
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features = self.extractor.from_file(row["audio_path"])
        target = torch.tensor(self.text.encode(row["transcript"]), dtype=torch.long)
        return features, target, self.text.normalize(row["transcript"])


def collate_batch(batch):
    features, targets, references = zip(*batch)
    padded_features, feature_lengths = pad_feature_batch(features)
    target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return padded_features, feature_lengths, targets, target_lengths, references


def run_epoch(model, loader, criterion, optimizer, device, text):
    is_training = optimizer is not None
    model.train(is_training)
    started_at = time.perf_counter()
    total_loss = 0.0
    hypotheses = []
    references = []
    for features, feature_lengths, targets, target_lengths, batch_refs in loader:
        features = features.to(device)
        feature_lengths = feature_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.set_grad_enabled(is_training):
            log_probs, output_lengths = model(features, feature_lengths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item()
        predictions = log_probs.argmax(dim=-1).transpose(0, 1).detach().cpu().tolist()
        hypotheses.extend(text.decode_greedy(prediction) for prediction in predictions)
        references.extend(batch_refs)

    pairs = list(zip(references, hypotheses))
    sample_count = max(1, len(pairs))
    wer = sum(word_error_rate(ref, hyp) for ref, hyp in pairs) / sample_count
    cer = sum(char_error_rate(ref, hyp) for ref, hyp in pairs) / sample_count
    exact_match = sum(ref == hyp for ref, hyp in pairs) / sample_count
    return {
        "loss": total_loss / max(1, len(loader)),
        "wer": wer,
        "cer": cer,
        "exact_match": exact_match,
        "seconds": time.perf_counter() - started_at,
    }


def format_metrics(prefix, metrics):
    return (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_wer={metrics['wer']:.4f} "
        f"{prefix}_cer={metrics['cer']:.4f} "
        f"{prefix}_exact_match={metrics['exact_match']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--checkpoint", default="artifacts/model1.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = TextTransform()
    extractor = MfccExtractor(sample_rate=args.sample_rate, n_mfcc=args.n_mfcc)
    train_data = SpeechCsvDataset(args.train_csv, extractor, text)
    valid_data = SpeechCsvDataset(args.valid_csv, extractor, text)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    config = {"n_mfcc": args.n_mfcc, "vocab_size": len(text.vocab)}
    model = CnnBiLstmCtc(**config).to(device)
    criterion = nn.CTCLoss(blank=text.blank_index, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_wer = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, text)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, criterion, None, device, text)
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(epoch_metrics)
        improved = valid_metrics["wer"] < best_wer
        print(
            f"epoch={epoch}/{args.epochs} "
            f"{format_metrics('train', train_metrics)} "
            f"{format_metrics('valid', valid_metrics)} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"epoch_seconds={train_metrics['seconds'] + valid_metrics['seconds']:.1f} "
            f"best={'yes' if improved else 'no'}",
            flush=True,
        )
        if improved:
            best_wer = valid_metrics["wer"]
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "vocab": text.vocab,
                    "sample_rate": args.sample_rate,
                    "best_wer": best_wer,
                    "history": history,
                },
                args.checkpoint,
            )


if __name__ == "__main__":
    main()
