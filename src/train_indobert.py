"""Fine-tuning IndoBERT untuk klasifikasi 3 kelas ujaran kebencian.

Desain eksperimen sengaja dibuat identik dengan `src/train.py` agar
perbandingan SVM vs IndoBERT bersifat apple-to-apple:

1. Sumber data dan fungsi pembersihan sama (`data/processed/dataset_clean.csv`).
2. Split sama: 80% train, 10% validation, 10% test, `random_state=42`.
3. Validation hanya dipakai untuk memilih jumlah epoch terbaik.
4. Setelah epoch terbaik diketahui, model dilatih ulang pada train+val
   (persis seperti SVM) lalu diuji SEKALI pada test.

Jalankan di GPU (Colab T4):

    pip install -q transformers
    python src/train_indobert.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset_clean.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"

RANDOM_STATE = 42
MODEL_NAME = "indobenchmark/indobert-base-p1"
LABELS = ["Abusive", "Hate Speech", "Normal"]
MAX_LEN = 96
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning IndoBERT 3 kelas.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="CSV berisi kolom text_final dan label.",
    )
    parser.add_argument("--tag", default="ibrohim", help="Sufiks nama artefak.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--weighting",
        choices=["inverse", "sqrt", "none"],
        default="sqrt",
        help=(
            "inverse = setara class_weight='balanced'; sqrt = versi lebih lunak "
            "(risiko menghafal kelas minoritas lebih kecil); none = tanpa bobot."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing untuk mengurangi overconfidence.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Berhenti awal bila val macro-F1 tidak naik selama N epoch (0 = mati).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="metrics.json pembanding (opsional).",
    )
    return parser.parse_args()


def compute_class_weights(series, device, mode: str, labels) -> torch.Tensor | None:
    """Bobot kelas untuk CrossEntropyLoss."""
    if mode == "none":
        return None
    total = len(series)
    raw = [total / (len(labels) * series[label]) for label in labels]
    if mode == "sqrt":
        raw = [math.sqrt(value) for value in raw]
    return torch.tensor(raw, dtype=torch.float32, device=device)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Kolom wajib: `text_final` dan `label`."""
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["text_final", "label"])
    frame = frame[frame["text_final"].str.strip().str.len() > 0]
    return frame.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, truths = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            losses.append(float(criterion(logits, labels)))
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            truths.extend(labels.cpu().numpy().tolist())
    macro_f1 = f1_score(truths, preds, average="macro")
    return float(np.mean(losses)), macro_f1, np.array(truths), np.array(preds)


def run_epochs(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor | None,
    tag: str,
    label_smoothing: float = 0.0,
    patience: int = 0,
    train_probe_loader=None,
) -> tuple[int, dict[str, float], float]:
    """Latih `epochs` epoch, kembalikan epoch terbaik menurut val macro-F1.

    `train_probe_loader` dipakai untuk mengukur macro-F1 pada sebagian data
    latih, supaya selisih train vs validation bisa dilaporkan sebagai bukti
    model tidak sekadar menghafal.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_epoch, best_f1, best_state, history = 1, -1.0, None, {}
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            scheduler.step()
            running += float(loss)
            if step % 100 == 0:
                print(f"  [{tag}] epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f}", flush=True)

        train_f1 = None
        if val_loader is None:
            # Fase 2 tidak punya validation set; evaluasi penuh tiap epoch cuma
            # membuang waktu tanpa memengaruhi keputusan apa pun.
            val_loss, val_f1 = float("nan"), -1.0
        else:
            val_loss, val_f1, _, _ = evaluate(model, val_loader, device)
            if train_probe_loader is not None:
                _, train_f1, _, _ = evaluate(model, train_probe_loader, device)
        elapsed = time.time() - started
        history[f"epoch_{epoch}"] = {
            "train_loss": round(running / len(train_loader), 4),
            "train_macro_f1_sample": None if train_f1 is None else round(train_f1, 4),
            "val_loss": None if val_loader is None else round(val_loss, 4),
            "val_macro_f1": None if val_loader is None else round(val_f1, 4),
            "generalization_gap": None if train_f1 is None else round(train_f1 - val_f1, 4),
            "seconds": round(elapsed, 1),
        }
        if val_loader is None:
            print(
                f"  [{tag}] epoch {epoch}: train_loss={running / len(train_loader):.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            continue
        gap = "" if train_f1 is None else f" train_f1={train_f1:.4f} gap={train_f1 - val_f1:+.4f}"
        print(
            f"  [{tag}] epoch {epoch}: train_loss={running / len(train_loader):.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}{gap} ({elapsed:.0f}s)",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                print(
                    f"  [{tag}] berhenti awal: val macro-F1 tidak naik "
                    f"{patience} epoch berturut-turut",
                    flush=True,
                )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, history, best_f1


def main() -> None:
    args = parse_args()

    # Argumen menimpa konstanta modul; semua pemakaian ada di dalam main().
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    data_path = Path(args.data)
    tag = args.tag
    metrics_out = ROOT / "results" / f"indobert_{tag}_metrics.json"
    report_out = ROOT / "results" / f"indobert_{tag}_classification_report.txt"
    model_out = ROOT / "models" / f"indobert_{tag}"

    set_seed()
    device = pick_device()
    print(f"Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model  : {MODEL_NAME}")
    print(f"Dataset: {data_path} (tag={tag})")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    frame = load_dataset(data_path)
    print(f"Total data: {len(frame):,}")

    # Label diambil dari data, jadi skrip ini bisa 2 kelas maupun 3 kelas.
    LABELS = sorted(frame["label"].unique().tolist())
    print(f"Kelas ({len(LABELS)}): {LABELS}")

    X = frame["text_final"]
    y = frame["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Split: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    for name, series in (("train", y_train), ("val", y_val), ("test", y_test)):
        counts = series.value_counts().to_dict()
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    to_ids = {label: index for index, label in enumerate(LABELS)}

    train_ds = TextDataset(X_train.tolist(), [to_ids[v] for v in y_train], tokenizer, MAX_LEN)
    val_ds = TextDataset(X_val.tolist(), [to_ids[v] for v in y_val], tokenizer, MAX_LEN)
    test_ds = TextDataset(X_test.tolist(), [to_ids[v] for v in y_test], tokenizer, MAX_LEN)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collator)

    # Bobot kelas: `sqrt` mencegah model menghafal kelas minoritas yang
    # jumlahnya hanya ratusan.
    weights = compute_class_weights(y_train.value_counts(), device, args.weighting, LABELS)
    print("Weighting:", args.weighting)
    if weights is not None:
        print("Class weights:", {label: round(float(w), 4) for label, w in zip(LABELS, weights)})

    # Sampel kecil data latih untuk mengukur selisih train vs validation.
    probe_size = min(2_000, len(train_ds))
    probe_indices = random.sample(range(len(train_ds)), probe_size)
    probe_loader = DataLoader(
        torch.utils.data.Subset(train_ds, probe_indices),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
    )

    def fresh_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS)
        ).to(device)
        # Checkpoint IndoBERT membawa id2label 5 kelas; timpa dengan label kita.
        model.config.id2label = {i: label for i, label in enumerate(LABELS)}
        model.config.label2id = {label: i for i, label in enumerate(LABELS)}
        return model

    # ---- Fase 1: pilih epoch terbaik di validation -------------------------
    print("\n=== Fase 1: latih di train, pantau validation ===")
    model = fresh_model()
    best_epoch, history, best_f1 = run_epochs(
        model,
        train_loader,
        val_loader,
        device,
        EPOCHS,
        LEARNING_RATE,
        weights,
        "train",
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        train_probe_loader=probe_loader,
    )
    print(f"Epoch terbaik: {best_epoch} (val macro-F1={best_f1:.4f})")

    # ---- Fase 2: latih ulang di train+val, uji sekali di test --------------
    print(f"\n=== Fase 2: latih ulang di train+val selama {best_epoch} epoch ===")
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    full_ds = TextDataset(X_full.tolist(), [to_ids[v] for v in y_full], tokenizer, MAX_LEN)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    weights_full = compute_class_weights(y_full.value_counts(), device, args.weighting, LABELS)
    final_model = fresh_model()
    run_epochs(
        final_model,
        full_loader,
        None,
        device,
        best_epoch,
        LEARNING_RATE,
        weights_full,
        "full",
        label_smoothing=args.label_smoothing,
    )

    # ---- Evaluasi test -----------------------------------------------------
    test_loss, test_f1, truths, preds = evaluate(final_model, test_loader, device)
    accuracy = accuracy_score(truths, preds)
    macro_f1 = f1_score(truths, preds, average="macro")
    matrix = confusion_matrix(truths, preds, labels=list(range(len(LABELS))))
    report = classification_report(
        truths, preds, labels=list(range(len(LABELS))), target_names=LABELS, digits=4
    )
    per_class = {}
    for index, label in enumerate(LABELS):
        mask_t = truths == index
        tp = int(((preds == index) & mask_t).sum())
        precision = tp / max(1, int((preds == index).sum()))
        recall = tp / max(1, int(mask_t.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(mask_t.sum()),
        }

    print("\n=== Hasil test IndoBERT ===")
    print(f"accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}")
    print(report)
    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    print(matrix)

    comparison = {}
    baseline_path = args.baseline
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        best = baseline.get("best_metrics", {})
        comparison = {
            "baseline_accuracy": best.get("accuracy"),
            "baseline_macro_f1": best.get("macro_f1"),
            "delta_accuracy": round(accuracy - float(best.get("accuracy", 0)), 4),
            "delta_macro_f1": round(macro_f1 - float(best.get("macro_f1", 0)), 4),
        }
        print("\n=== Perbandingan dengan baseline ===")
        print(json.dumps(comparison, indent=2))

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": MODEL_NAME,
        "model_type": "indobert",
        "device": str(device),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_tried": EPOCHS,
        "best_epoch": best_epoch,
        "weighting": args.weighting,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "validation_macro_f1": round(best_f1, 4),
        "test_loss": round(test_loss, 4),
        "test_size": int(len(y_test)),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "history": history,
        "comparison_to_baseline": comparison,
        "dataset": str(data_path.relative_to(ROOT)) if str(data_path).startswith(str(ROOT)) else str(data_path),
        "tag": tag,
        "use_stemming": False,
        "input_column": "text_final",
        "split": {"random_state": RANDOM_STATE, "test_size": 0.1, "val_size": 0.1},
    }
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_out.write_text(
        f"IndoBERT ({MODEL_NAME}) - dataset {data_path.name}\n"
        f"best_epoch={best_epoch}  val_macro_f1={best_f1:.4f}\n"
        f"test_accuracy={accuracy:.4f}  test_macro_f1={macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix:\n{matrix}\n",
        encoding="utf-8",
    )

    model_out.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_out, safe_serialization=True)
    tokenizer.save_pretrained(model_out)
    (model_out / "label_map.json").write_text(
        json.dumps(to_ids, indent=2), encoding="utf-8"
    )

    print(f"\nMetrik  : {metrics_out}")
    print(f"Laporan : {report_out}")
    print(f"Model   : {model_out}")


if __name__ == "__main__":
    main()
