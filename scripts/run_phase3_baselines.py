"""
Phase 3: Run and Record All Trivial and TF-IDF Baselines Across 15 Outer Folds.

Models evaluated:
1. most_frequent
2. first_token_prior
3. tfidf_knn (fold-isolated vectorizer, inner-tuned tau)
4. tfidf_logistic (fold-isolated vectorizer, balanced OvR, inner-tuned tau)

Appends all runs to ARN/results/ledger.json with exact k/N counts.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.baselines import (
    MostFrequentBaseline,
    FirstTokenPriorBaseline,
    TfidfKnnBaseline,
    TfidfLogisticBaseline,
)
from arn.tokenizer import TOOL_TO_IDX, IDX_TO_TOOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.phase3")


def get_git_hash(repo_dir: Path) -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    N = len(y_true)
    matches = np.all(y_true == y_pred, axis=1)
    k_subset = int(matches.sum())
    subset_acc = float(k_subset / N)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "tool_subset_acc": {"k": k_subset, "N": N, "rate": subset_acc},
        "tool_micro_f1": micro_f1,
        "tool_macro_f1": macro_f1,
        "strict_span_f1": 0.0,
        "lenient_span_f1": 0.0,
        "joint_em": {"k": 0, "N": N, "rate": 0.0},
        "seen_tool_subset_acc": {"k": k_subset, "N": N, "rate": subset_acc},
        "seen_joint_em": {"k": 0, "N": N, "rate": 0.0},
    }


def run_phase3_baselines():
    repo_dir = Path(__file__).resolve().parent.parent
    manifest_path = repo_dir / "data" / "splits" / "cv_manifest.json"
    raw_path = repo_dir / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    ledger_path = repo_dir / "results" / "ledger.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    user_texts = [next(m["content"] for m in s["messages"] if m["role"] == "user") for s in raw_samples]
    num_samples = len(raw_samples)
    num_tools = len(TOOL_TO_IDX)

    tool_targets = np.zeros((num_samples, num_tools), dtype=np.float32)
    for i, s in enumerate(raw_samples):
        asst = next(m["content"] for m in s["messages"] if m["role"] == "assistant")
        for c in json.loads(asst):
            tool_targets[i, TOOL_TO_IDX[c["name"]]] = 1.0

    git_hash = get_git_hash(repo_dir)
    timestamp = datetime.utcnow().isoformat() + "Z"
    command = "python scripts/run_phase3_baselines.py"

    print("=== RUNNING PHASE 3 BASELINES ACROSS 15 OUTER FOLDS ===", flush=True)

    baseline_names = ["most_frequent", "first_token_prior", "tfidf_knn", "tfidf_logistic"]
    runs_added = 0

    for sh_name, sh_folds in manifest["folds"].items():
        seed = {"shuffle_1": 42, "shuffle_2": 100, "shuffle_3": 2024}.get(sh_name, 42)
        print(f"\n--- {sh_name} (seed={seed}) ---", flush=True)

        for fold_name, f_data in sh_folds.items():
            tr_idx = f_data["outer_train_indices"]
            val_idx = f_data["outer_val_indices"]
            in_tr_idx = f_data["inner_train_indices"]
            in_val_idx = f_data["inner_val_indices"]

            tr_texts = [user_texts[i] for i in tr_idx]
            val_texts = [user_texts[i] for i in val_idx]
            in_tr_texts = [user_texts[i] for i in in_tr_idx]
            in_val_texts = [user_texts[i] for i in in_val_idx]

            y_tr = tool_targets[tr_idx]
            y_val = tool_targets[val_idx]
            y_in_tr = tool_targets[in_tr_idx]
            y_in_val = tool_targets[in_val_idx]

            # 1. Most Frequent
            mf = MostFrequentBaseline(num_tools=num_tools).fit(tr_texts, y_tr)
            mf_pred = mf.predict(val_texts)
            mf_metrics = compute_multilabel_metrics(y_val, mf_pred)

            # 2. First Token Prior
            ftp = FirstTokenPriorBaseline(num_tools=num_tools).fit(tr_texts, y_tr)
            ftp_pred = ftp.predict(val_texts, threshold=0.50)
            ftp_metrics = compute_multilabel_metrics(y_val, ftp_pred)

            # 3. TF-IDF kNN (tune tau on inner val)
            knn_in = TfidfKnnBaseline(n_neighbors=5, min_df=2).fit(in_tr_texts, y_in_tr)
            best_tau_knn = knn_in.tune_threshold(in_val_texts, y_in_val)
            knn_full = TfidfKnnBaseline(n_neighbors=5, min_df=2).fit(tr_texts, y_tr)
            knn_pred = knn_full.predict(val_texts, tau=best_tau_knn)
            knn_metrics = compute_multilabel_metrics(y_val, knn_pred)
            knn_metrics["best_tau"] = best_tau_knn

            # 4. TF-IDF LogReg (balanced, tune tau on inner val)
            lr_in = TfidfLogisticBaseline(C=1.0, max_iter=300, min_df=2).fit(in_tr_texts, y_in_tr)
            best_tau_lr = lr_in.tune_threshold(in_val_texts, y_in_val)
            lr_full = TfidfLogisticBaseline(C=1.0, max_iter=300, min_df=2).fit(tr_texts, y_tr)
            lr_pred = lr_full.predict(val_texts, tau=best_tau_lr)
            lr_metrics = compute_multilabel_metrics(y_val, lr_pred)
            lr_metrics["best_tau"] = best_tau_lr

            models_results = [
                ("most_frequent", mf_metrics),
                ("first_token_prior", ftp_metrics),
                ("tfidf_knn", knn_metrics),
                ("tfidf_logistic", lr_metrics),
            ]

            for m_name, m_metrics in models_results:
                exp_id = f"p3_{m_name}_{sh_name}_{fold_name}"
                run_record = {
                    "experiment_id": exp_id,
                    "model": m_name,
                    "protocol": "phase3_baselines",
                    "shuffle": sh_name,
                    "fold": fold_name,
                    "seed": seed,
                    "metrics": m_metrics,
                    "git_hash": git_hash,
                    "command": command,
                    "timestamp": timestamp,
                }
                ledger["runs"].append(run_record)
                runs_added += 1

            print(
                f"  {fold_name:8s} | "
                f"MF Acc: {mf_metrics['tool_subset_acc']['rate']*100:5.2f}% | "
                f"FirstTok Acc: {ftp_metrics['tool_subset_acc']['rate']*100:5.2f}% | "
                f"kNN Acc: {knn_metrics['tool_subset_acc']['rate']*100:5.2f}% (tau={best_tau_knn:.2f}) | "
                f"LogReg Acc: {lr_metrics['tool_subset_acc']['rate']*100:5.2f}% (tau={best_tau_lr:.2f})"
            )

    # Save ledger
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    print(f"\nSaved {runs_added} baseline experiment records to {ledger_path}", flush=True)


if __name__ == "__main__":
    run_phase3_baselines()
