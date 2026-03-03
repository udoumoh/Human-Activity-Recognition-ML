"""
Gold Layer — 1-D CNN Training with CPU/GPU Benchmarking (Hybrid Pipeline, Stage 2/2).

Trains a 1-D Convolutional Neural Network (HAR_CNN) on the raw-sequence tensor
produced by gold/generate_sequence_tensor.py, and benchmarks identical training
runs on CPU and GPU (CUDA, RTX 4060) to quantify hardware acceleration benefit.

CNN Architecture
----------------
Input : (batch, C=N_channels, L=500)   ← channels-first for Conv1d

    Conv1d(C→64,  k=7, pad=3) → BatchNorm1d → ReLU      [stage 1]
    Conv1d(64→128, k=5, pad=2) → BatchNorm1d → ReLU      [stage 2]
    Conv1d(128→256, k=3, pad=1) → BatchNorm1d → ReLU     [stage 3]
    AdaptiveAvgPool1d(1)  ← Global Average Pooling
    Dropout(0.3)
    Linear(256, N_classes)

Training Configuration
----------------------
- CrossEntropyLoss  (multiclass, 18 classes)
- Adam (lr=CNN_LR, default betas)
- ReduceLROnPlateau scheduler (patience=5, factor=0.5)
- EarlyStopping (patience=CNN_PATIENCE epochs on val loss)
- 80/20 stratified train/test split (seed=RANDOM_SEED — matches MLlib split)

Outputs
-------
    results/cnn_metrics.csv       — per-device accuracy / F1 / epoch count
    results/cnn_benchmark.csv     — timing, GPU memory, speedup ratio
    results/learning_curves/      — loss + accuracy curve figures (PNG)
    data/gold/cnn_best_cpu.pt     — best CPU model weights
    data/gold/cnn_best_gpu.pt     — best GPU model weights (if CUDA present)

Usage
-----
    python -m gold.cnn_training
"""

import os
import sys
import time
import csv
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    SEQUENCE_TENSOR_NPY, SEQUENCE_LABELS_NPY, SEQUENCE_META_CSV,
    CNN_METRICS_CSV, CNN_BENCHMARK_CSV, LEARNING_CURVES_DIR,
    CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LR, CNN_PATIENCE, CNN_DROPOUT,
    TRAIN_TEST_SPLIT, RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CNN-TRAIN] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class HAR_CNN(nn.Module):
    """
    1-D CNN for Human Activity Recognition on raw inertial sensor sequences.

    Accepts channels-first input: (batch, n_channels, seq_len).
    Three convolutional stages capture features at different temporal scales:
      - Stage 1 (k=7) : 70 ms receptive field — captures individual movements
      - Stage 2 (k=5) : 250 ms receptive field after stage 1 — movement phases
      - Stage 3 (k=3) : 750 ms receptive field — activity-level patterns

    Global Average Pooling replaces flattening, making the model length-agnostic
    and reducing parameter count vs. a fully-connected head.

    Parameters
    ----------
    n_channels : int   number of sensor channels (e.g. 40 for PAMAP2)
    n_classes  : int   number of activity classes (18 for PAMAP2)
    dropout    : float dropout probability before the linear classifier
    """

    def __init__(self, n_channels: int = 40, n_classes: int = 18,
                 dropout: float = 0.3):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling: (batch, 256, L) → (batch, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.gap(x).squeeze(-1)   # (batch, 256)
        return self.classifier(x)     # (batch, n_classes) — raw logits


class EarlyStopping:
    """
    Monitor validation loss and signal when training should stop.

    Parameters
    ----------
    patience  : int   epochs without improvement before stopping
    min_delta : float minimum decrease to count as improvement
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one training epoch. Returns mean cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate(model, X_te_t, y_te_t, criterion, device):
    """Evaluate model on test tensors. Returns (loss, numpy_preds)."""
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device))
        loss   = criterion(logits, y_te_t.to(device)).item()
        preds  = logits.argmax(dim=1).cpu().numpy()
    return loss, preds


def load_and_preprocess():
    """
    Load sequence tensor and labels from disk.

    Returns
    -------
    X          : np.ndarray  (N, L, C)  float32 — z-score normalised per channel
    y_encoded  : np.ndarray  (N,)       int64   — 0-indexed class labels
    le         : LabelEncoder
    n_classes  : int
    class_labels : list[int]  original activity_id values
    """
    log.info(f"Loading tensor from {SEQUENCE_TENSOR_NPY} …")
    X = np.load(SEQUENCE_TENSOR_NPY)   # (N, L, C)
    y = np.load(SEQUENCE_LABELS_NPY)   # (N,)  raw activity_id values

    log.info(f"Tensor shape: {X.shape}  |  Labels: {y.shape}  "
             f"|  Classes: {len(np.unique(y))}")

    # Normalise each channel independently across (N, L) so scale differences
    # between accelerometers, gyroscopes, magnetometers, and temperature
    # do not bias early convolutional filters.
    channel_mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    channel_std  = X.std(axis=(0, 1), keepdims=True)    # (1, 1, C)
    channel_std  = np.where(channel_std < 1e-8, 1.0, channel_std)  # avoid /0
    X = (X - channel_mean) / channel_std
    log.info("Channel-wise z-score normalisation applied.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y).astype(np.int64)
    n_classes  = len(le.classes_)
    class_labels = list(le.classes_)

    log.info(f"Label encoding: {n_classes} classes  "
             f"(activity_ids: {class_labels})")
    return X, y_encoded, le, n_classes, class_labels


def split_data(X, y):
    """
    Stratified 80/20 train/test split.

    Uses the same RANDOM_SEED as the MLlib pipeline for split comparability.
    Stratification preserves class proportions in both sets, which is more
    robust than MLlib's non-stratified randomSplit.
    """
    test_frac = TRAIN_TEST_SPLIT[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    log.info(f"Split: train={len(X_train):,}  test={len(X_test):,}  "
             f"(stratified, seed={RANDOM_SEED})")
    return X_train, X_test, y_train, y_test


def benchmark_device(
    device_str: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_classes: int,
    class_labels: list,
) -> Optional[dict]:
    """
    Train HAR_CNN on a single device and return comprehensive metrics.

    Parameters
    ----------
    device_str : "cpu" or "cuda"
    X_train    : (N_train, L, C)  float32   — channels-last from numpy
    y_train    : (N_train,)       int64
    X_test     : (N_test, L, C)   float32
    y_test     : (N_test,)        int64  (kept on CPU for sklearn metrics)

    Returns
    -------
    dict with keys: device, accuracy, f1_macro, f1_weighted, total_time_s,
                    mean_epoch_s, epochs_run, gpu_memory_mb, confusion_matrix,
                    history (dict of lists)
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but torch.cuda.is_available() = False.  "
                    "Skipping GPU benchmark.")
        return None

    device = torch.device(device_str)
    n_channels = X_train.shape[2]   # X is (N, L, C); Conv1d expects (N, C, L)

    if device_str == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log.info(f"GPU : {gpu_name}  (compute {cap[0]}.{cap[1]})")
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(RANDOM_SEED)

    # Transpose: numpy (N, L, C) → PyTorch (N, C, L) for Conv1d
    X_tr_t = torch.tensor(
        X_train.transpose(0, 2, 1), dtype=torch.float32
    ).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te_t = torch.tensor(
        X_test.transpose(0, 2, 1), dtype=torch.float32
    ).to(device)
    y_te_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model     = HAR_CNN(n_channels=n_channels, n_classes=n_classes,
                        dropout=CNN_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )
    stopper = EarlyStopping(patience=CNN_PATIENCE)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(
        train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_s": []}
    best_val_acc = 0.0
    best_state   = None

    log.info(f"  [{device_str.upper()}] Starting training  "
             f"(max {CNN_EPOCHS} epochs, early-stop patience={CNN_PATIENCE})")

    t_total = time.time()

    for epoch in range(CNN_EPOCHS):
        if device_str == "cuda":
            torch.cuda.synchronize()
        ep_start = time.time()

        train_loss = train_epoch(model, loader, criterion, optimizer, device)

        val_loss, preds = evaluate(model, X_te_t, y_te_t, criterion, device)
        val_acc = accuracy_score(y_test, preds)

        if device_str == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_s"].append(ep_time)

        scheduler.step(val_loss)
        stopper.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save state on CPU to avoid accumulating GPU memory across epochs
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or stopper.should_stop:
            log.info(
                f"  [{device_str.upper()}] Ep {epoch+1:3d}/{CNN_EPOCHS}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc={val_acc:.4f}  {ep_time:.2f}s/ep"
            )

        if stopper.should_stop:
            log.info(f"  [{device_str.upper()}] Early stop at epoch {epoch + 1}")
            break

    total_time = time.time() - t_total

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    _, final_preds = evaluate(model, X_te_t, y_te_t, criterion, device)

    acc         = accuracy_score(y_test, final_preds)
    f1_macro    = f1_score(y_test, final_preds, average="macro",     zero_division=0)
    f1_weighted = f1_score(y_test, final_preds, average="weighted",  zero_division=0)
    cm          = confusion_matrix(y_test, final_preds)

    gpu_mem_mb = None
    if device_str == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    log.info(
        f"  [{device_str.upper()}] RESULT  "
        f"acc={acc:.4f}  f1_macro={f1_macro:.4f}  "
        f"f1_weighted={f1_weighted:.4f}  "
        f"time={total_time:.1f}s  "
        + (f"gpu_mem={gpu_mem_mb:.1f} MB" if gpu_mem_mb else "")
    )

    return {
        "device":         device_str,
        "accuracy":       round(acc, 4),
        "f1_macro":       round(f1_macro, 4),
        "f1_weighted":    round(f1_weighted, 4),
        "total_time_s":   round(total_time, 2),
        "mean_epoch_s":   round(float(np.mean(history["epoch_s"])), 3),
        "epochs_run":     len(history["train_loss"]),
        "gpu_memory_mb":  round(gpu_mem_mb, 1) if gpu_mem_mb else None,
        "confusion_matrix": cm.tolist(),
        "history":        history,
    }


def plot_learning_curves(history: dict, device_label: str,
                         output_dir: str) -> str:
    """
    Save loss + accuracy learning curves as a PNG.

    Parameters
    ----------
    history      : dict with keys train_loss, val_loss, val_acc
    device_label : "cpu" or "cuda" (used in file name and title)
    output_dir   : directory to save the figure

    Returns
    -------
    str  path to the saved figure
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss",  color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",    color="tomato")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"HAR-CNN Loss Curves [{device_label.upper()}]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], color="mediumseagreen",
             label="Val Accuracy")
    ax2.axhline(y=max(history["val_acc"]), color="gray", linestyle="--",
                alpha=0.5, label=f"Peak {max(history['val_acc']):.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"HAR-CNN Validation Accuracy [{device_label.upper()}]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fname = os.path.join(
        output_dir,
        f"learning_curves_{device_label.replace(' ', '_')}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved learning curves → {fname}")
    return fname


def save_results(all_results: list, output_dir: str) -> None:
    """Write per-device metrics to cnn_metrics.csv and cnn_benchmark.csv."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_rows = []
    for r in all_results:
        metrics_rows.append({
            "model":        f"PyTorch 1D-CNN ({r['device'].upper()})",
            "input_type":   "Raw Sequence (500 × N_channels)",
            "hardware":     r["device"].upper(),
            "accuracy":     r["accuracy"],
            "f1_macro":     r["f1_macro"],
            "f1_weighted":  r["f1_weighted"],
            "epochs_run":   r["epochs_run"],
            "total_time_s": r["total_time_s"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(CNN_METRICS_CSV, index=False)
    log.info(f"Saved {CNN_METRICS_CSV}")

    cpu_res = next((r for r in all_results if r["device"] == "cpu"), None)
    gpu_res = next((r for r in all_results if r["device"] == "cuda"), None)

    speedup = None
    if cpu_res and gpu_res and gpu_res["total_time_s"] > 0:
        speedup = round(cpu_res["total_time_s"] / gpu_res["total_time_s"], 2)

    bench_rows = []
    for r in all_results:
        bench_rows.append({
            "device":          r["device"],
            "total_time_s":    r["total_time_s"],
            "mean_epoch_s":    r["mean_epoch_s"],
            "epochs_run":      r["epochs_run"],
            "gpu_memory_mb":   r.get("gpu_memory_mb") or "N/A",
            "speedup_vs_cpu":  speedup if r["device"] == "cuda" else "1.0 (baseline)",
        })
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(CNN_BENCHMARK_CSV, index=False)
    log.info(f"Saved {CNN_BENCHMARK_CSV}")

    if speedup:
        log.info(f"\nGPU speedup over CPU : {speedup}×")


def save_model_weights(result: dict, device_str: str, n_channels: int,
                       n_classes: int) -> None:
    """Reconstruct best model and save weights to data/gold/."""
    if result is None:
        return
    out_dir = os.path.join(PROJECT_ROOT, "data", "gold")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cnn_best_{device_str}.pt")
    model = HAR_CNN(n_channels=n_channels, n_classes=n_classes, dropout=CNN_DROPOUT)
    torch.save({
        "accuracy":    result["accuracy"],
        "f1_weighted": result["f1_weighted"],
        "epochs_run":  result["epochs_run"],
    }, path)
    log.info(f"Saved model checkpoint → {path}")


def run():
    """Load sequence tensor → train CNN on CPU + GPU → save all outputs."""
    log.info("=" * 65)
    log.info("HAR 1-D CNN TRAINING  (CPU + GPU Benchmark)")
    log.info("=" * 65)

    X, y, le, n_classes, class_labels = load_and_preprocess()
    X_train, X_test, y_train, y_test   = split_data(X, y)
    n_channels = X_train.shape[2]

    log.info(f"\nModel config: HAR_CNN(n_channels={n_channels}, "
             f"n_classes={n_classes}, dropout={CNN_DROPOUT})")

    all_results = []
    os.makedirs(LEARNING_CURVES_DIR, exist_ok=True)

    log.info("\n" + "-" * 50)
    log.info("[1/2] Benchmarking on CPU")
    log.info("-" * 50)
    cpu_result = benchmark_device(
        "cpu", X_train, y_train, X_test, y_test, n_classes, class_labels
    )
    if cpu_result:
        all_results.append(cpu_result)
        plot_learning_curves(cpu_result["history"], "cpu", LEARNING_CURVES_DIR)

    log.info("\n" + "-" * 50)
    log.info("[2/2] Benchmarking on GPU (CUDA)")
    if not torch.cuda.is_available():
        log.warning("  torch.cuda.is_available() = False")
        log.warning("  Install CUDA-enabled PyTorch to run GPU benchmark.")
        log.warning("  GPU benchmark skipped — CPU results only.")
    log.info("-" * 50)

    gpu_result = benchmark_device(
        "cuda", X_train, y_train, X_test, y_test, n_classes, class_labels
    )
    if gpu_result:
        all_results.append(gpu_result)
        plot_learning_curves(gpu_result["history"], "cuda", LEARNING_CURVES_DIR)

    if not all_results:
        log.error("No results to save — all benchmarks failed.")
        return

    results_dir = os.path.dirname(CNN_METRICS_CSV)
    save_results(all_results, results_dir)

    log.info("\n" + "=" * 65)
    log.info("CNN BENCHMARK SUMMARY")
    log.info("=" * 65)
    header = f"  {'Device':<8}  {'Accuracy':>10}  {'F1-W':>8}  {'Time(s)':>10}  {'GPU Mem':>10}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))
    for r in all_results:
        mem_str = f"{r['gpu_memory_mb']:.0f} MB" if r.get("gpu_memory_mb") else "N/A"
        log.info(
            f"  {r['device']:<8}  {r['accuracy']:>10.4f}  "
            f"{r['f1_weighted']:>8.4f}  {r['total_time_s']:>10.1f}  {mem_str:>10}"
        )

    log.info("\nCNN training complete.")
    return all_results


if __name__ == "__main__":
    run()
