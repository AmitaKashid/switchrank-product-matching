import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

sns.set_theme(style="whitegrid", palette="muted")

def compute_all_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    hard_neg_mask: np.ndarray = None,
    latencies_ms: List[float] = None,
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)

    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    p_arr, r_arr, _ = precision_recall_curve(y_true, probs)
    pr_auc = float(auc(r_arr, p_arr))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))

    hard_neg_fpr = 0.0
    if hard_neg_mask is not None and np.sum(hard_neg_mask) > 0:
        hn_preds = preds[hard_neg_mask]
        hn_true = y_true[hard_neg_mask]
        hn_fp = np.sum((hn_preds == 1) & (hn_true == 0))
        hn_tn = np.sum((hn_preds == 0) & (hn_true == 0))
        hard_neg_fpr = float(hn_fp / max(hn_fp + hn_tn, 1))

    p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "hard_neg_fpr": hard_neg_fpr,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
    }

def plot_pr_curves(eval_results: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path):
    plt.figure(figsize=(7, 5))
    for name, (y_true, probs) in eval_results.items():
        p, r, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(r, p)
        plt.plot(r, p, label=f"{name} (AUC = {pr_auc:.3f})", lw=2)

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves Across Baseline & Models", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_reliability_diagram(y_true: np.ndarray, probs_uncal: np.ndarray, probs_cal: np.ndarray, out_path: Path):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6, 5))

    prob_true_un, prob_pred_un = calibration_curve(y_true, probs_uncal, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, probs_cal, n_bins=10)

    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.plot(prob_pred_un, prob_true_un, "s-", label="Uncalibrated LightGBM")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", label="Isotonic Calibrated LightGBM")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_precision_coverage(coverages: List[float], precisions: List[float], out_path: Path):
    plt.figure(figsize=(6, 5))
    plt.plot(coverages, precisions, "o-", color="#2b5c8f", lw=2)
    plt.axhline(0.99, color="crimson", linestyle="--", label="99% Precision Target")
    plt.xlabel("Auto-Match Coverage %", fontsize=11)
    plt.ylabel("Auto-Match Precision", fontsize=11)
    plt.title("Selective Decision Policy: Precision vs Coverage", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_healthcare_transfer(difficulty_metrics: Dict[str, Dict[str, float]], out_path: Path):
    diffs = list(difficulty_metrics.keys())
    accs = [difficulty_metrics[d]["accuracy"] for d in diffs]
    f1s = [difficulty_metrics[d]["f1"] for d in diffs]

    x = np.arange(len(diffs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rects1 = ax.bar(x - width/2, accs, width, label="Top-1 Resolution Accuracy", color="#3470a3")
    rects2 = ax.bar(x + width/2, f1s, width, label="Pair F1 Score", color="#e76f51")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("AccessGUDID Healthcare Domain Transfer Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(diffs)
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
