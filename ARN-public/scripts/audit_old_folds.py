"""
Performs complete coverage audit of old Stage 2d folds across all 3 shuffles (15 folds).
Reports:
  - Per outer fold and per tool: training clusters and samples, validation clusters and samples.
  - Tools completely absent from training.
  - Share of validation samples containing unseen tools.
Saves results/coverage_audit_old_folds.json.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2_master_suite import partition_clusters


def audit_old_folds():
    train_path = Path("ARN/data/splits/group_disjoint_train.jsonl")
    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")

    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    frame_to_cluster = build_canonical_clusters(raw_path)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
    dataset = ARNDataset(str(train_path), tokenizer=TOKENIZER)

    # Extract active tools
    sample_tools = []
    for item in dataset.encoded:
        active = [IDX_TO_TOOL[i] for i, val in enumerate(item["tool_labels"].tolist()) if val > 0.5]
        sample_tools.append(active)

    shuffles = [
        {"name": "shuffle_1", "seed": 42},
        {"name": "shuffle_2", "seed": 100},
        {"name": "shuffle_3", "seed": 2024},
    ]

    all_tools = sorted(list(set(t for tools in sample_tools for t in tools)))
    audit_results = {}

    for sh in shuffles:
        sh_name = sh["name"]
        seed = sh["seed"]
        folds = partition_clusters(cluster_ids, n_folds=5, seed=seed)
        sh_audit = {}

        for f_idx in range(5):
            val_indices = set(folds[f_idx])
            train_indices = set(i for other_f in range(5) if other_f != f_idx for i in folds[other_f])

            tr_samples_by_tool = defaultdict(int)
            val_samples_by_tool = defaultdict(int)
            tr_clusters_by_tool = defaultdict(set)
            val_clusters_by_tool = defaultdict(set)

            for i in train_indices:
                cid = cluster_ids[i]
                for t in sample_tools[i]:
                    tr_samples_by_tool[t] += 1
                    tr_clusters_by_tool[t].add(cid)

            for i in val_indices:
                cid = cluster_ids[i]
                for t in sample_tools[i]:
                    val_samples_by_tool[t] += 1
                    val_clusters_by_tool[t].add(cid)

            absent_tools = [t for t in all_tools if tr_samples_by_tool[t] == 0]
            unseen_val_samples = sum(1 for i in val_indices if any(tr_samples_by_tool[t] == 0 for t in sample_tools[i]))
            share_unseen = (unseen_val_samples / len(val_indices)) * 100.0

            tool_breakdown = {}
            for t in all_tools:
                tool_breakdown[t] = {
                    "train_samples": tr_samples_by_tool[t],
                    "train_clusters": len(tr_clusters_by_tool[t]),
                    "val_samples": val_samples_by_tool[t],
                    "val_clusters": len(val_clusters_by_tool[t]),
                    "is_absent_from_train": tr_samples_by_tool[t] == 0,
                }

            fold_name = f"fold_{f_idx + 1}"
            sh_audit[fold_name] = {
                "train_sample_count": len(train_indices),
                "val_sample_count": len(val_indices),
                "tools_absent_from_train": absent_tools,
                "unseen_tool_val_sample_count": unseen_val_samples,
                "share_unseen_val_samples_pct": round(share_unseen, 2),
                "tool_breakdown": tool_breakdown,
            }

        audit_results[sh_name] = sh_audit

    out_file = Path("ARN/results/coverage_audit_old_folds.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Saved complete coverage audit to {out_file}")

    # Summary table
    print("\n=== SUMMARY OF OLD FOLD COVERAGE DEFECTS ===")
    print(f"{'Shuffle':12s} | {'Fold':8s} | {'Val N':6s} | {'Absent Tools':35s} | {'Unseen Val Samples':20s}")
    print("-" * 90)
    for sh_name, sh_data in audit_results.items():
        for fold_name, f_data in sh_data.items():
            absent_str = ", ".join(f_data["tools_absent_from_train"]) if f_data["tools_absent_from_train"] else "None"
            unseen_str = f"{f_data['unseen_tool_val_sample_count']}/{f_data['val_sample_count']} ({f_data['share_unseen_val_samples_pct']}%)"
            print(f"{sh_name:12s} | {fold_name:8s} | {f_data['val_sample_count']:6d} | {absent_str:35s} | {unseen_str:20s}")


if __name__ == "__main__":
    audit_old_folds()
