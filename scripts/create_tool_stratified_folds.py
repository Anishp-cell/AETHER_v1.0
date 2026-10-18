"""
Tool-Stratified Grouped K-Fold Generator for ARN (Phase 1).

Generates 15 outer folds (5 folds x 3 shuffles) over ALL 5,000 samples.
Constraints:
  1. Group-disjoint: Clusters are never split between train and val.
  2. Coverage guarantee: Every tool (except open_url, which is train-only)
     has >= 1 training cluster in every outer fold AND in every inner fold.
  3. open_url (1 cluster, 37 samples) is train-only across all folds.
  4. Near-duplicate check: Asserts max token-edit similarity between val and train skeletons < 0.80.
  5. Inner validation: 80/20 cluster-disjoint split inside each outer training fold.

Outputs:
  - ARN/data/splits/cv_manifest.json (with SHA-256 hashes)
  - ARN/results/coverage_audit_new_folds.json
"""

import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters, token_edit_sim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.splits")


def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()


def build_tool_stratified_folds(
    cluster_ids: List[str],
    sample_tools: List[List[str]],
    raw_skeletons: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[List[List[int]], List[int]]:
    """
    Partitions 5,000 samples into 5 tool-stratified, group-disjoint folds.
    Returns (outer_val_folds, train_only_indices).
    """
    rng = random.Random(seed)
    
    cluster_to_indices = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(idx)

    cluster_tools = defaultdict(set)
    for cid, idxs in cluster_to_indices.items():
        for i in idxs:
            cluster_tools[cid].update(sample_tools[i])

    # 1. Identify open_url cluster (train-only)
    train_only_indices = []
    regular_clusters = []
    for cid, tools in cluster_tools.items():
        if tools == {"open_url"} or (len(tools) == 1 and "open_url" in tools):
            train_only_indices.extend(cluster_to_indices[cid])
        else:
            regular_clusters.append(cid)

    # 2. Tool to clusters mapping
    tool_to_cids = defaultdict(list)
    for cid in regular_clusters:
        for t in cluster_tools[cid]:
            tool_to_cids[t].append(cid)

    sorted_tools = sorted(tool_to_cids.keys(), key=lambda t: (len(tool_to_cids[t]), t))

    assigned_cluster_fold = {}
    fold_sample_counts = [0] * n_folds
    fold_clusters = [set() for _ in range(n_folds)]
    fold_tool_clusters = [defaultdict(int) for _ in range(n_folds)]

    # Assign clusters of rarest tools first
    for t in sorted_tools:
        cids = [c for c in tool_to_cids[t] if c not in assigned_cluster_fold]
        rng.shuffle(cids)
        for cid in cids:
            eligible_folds = []
            for f in range(n_folds):
                can_assign = True
                for tool_name in cluster_tools[cid]:
                    total_cl = len(tool_to_cids[tool_name])
                    curr_in_f = fold_tool_clusters[f][tool_name]
                    if curr_in_f + 1 >= total_cl:
                        can_assign = False
                        break
                if can_assign:
                    eligible_folds.append(f)

            if not eligible_folds:
                best_fold = int(np.argmin(fold_sample_counts))
            else:
                best_fold = min(eligible_folds, key=lambda f: fold_sample_counts[f])

            assigned_cluster_fold[cid] = best_fold
            fold_clusters[best_fold].add(cid)
            fold_sample_counts[best_fold] += len(cluster_to_indices[cid])
            for tool_name in cluster_tools[cid]:
                fold_tool_clusters[best_fold][tool_name] += 1

    # Remaining clusters
    remaining_cids = [c for c in regular_clusters if c not in assigned_cluster_fold]
    rng.shuffle(remaining_cids)
    remaining_cids.sort(key=lambda c: len(cluster_to_indices[c]), reverse=True)

    for cid in remaining_cids:
        eligible_folds = []
        for f in range(n_folds):
            can_assign = True
            for tool_name in cluster_tools[cid]:
                total_cl = len(tool_to_cids[tool_name])
                curr_in_f = fold_tool_clusters[f][tool_name]
                if curr_in_f + 1 >= total_cl:
                    can_assign = False
                    break
            if can_assign:
                eligible_folds.append(f)

        if not eligible_folds:
            best_fold = int(np.argmin(fold_sample_counts))
        else:
            best_fold = min(eligible_folds, key=lambda f: fold_sample_counts[f])

        assigned_cluster_fold[cid] = best_fold
        fold_clusters[best_fold].add(cid)
        fold_sample_counts[best_fold] += len(cluster_to_indices[cid])
        for tool_name in cluster_tools[cid]:
            fold_tool_clusters[best_fold][tool_name] += 1

    outer_val_folds = [[] for _ in range(n_folds)]
    for cid, f in assigned_cluster_fold.items():
        outer_val_folds[f].extend(cluster_to_indices[cid])

    return outer_val_folds, train_only_indices


def partition_inner_split(
    train_indices: List[int],
    cluster_ids: List[str],
    sample_tools: List[List[str]],
    all_tools: List[str],
    rng: random.Random,
    val_ratio: float = 0.20,
) -> Tuple[List[int], List[int]]:
    """Partitions outer training indices into inner train (80%) and inner val (20%) preserving tool coverage."""
    cluster_to_indices = defaultdict(list)
    for idx in train_indices:
        cluster_to_indices[cluster_ids[idx]].append(idx)

    cluster_tools = defaultdict(set)
    for cid, idxs in cluster_to_indices.items():
        for i in idxs:
            cluster_tools[cid].update(sample_tools[i])

    tool_to_cids = defaultdict(list)
    for cid in cluster_to_indices:
        for t in cluster_tools[cid]:
            tool_to_cids[t].append(cid)

    must_be_train = set()
    for t in all_tools:
        cids = tool_to_cids[t]
        if len(cids) == 1:
            must_be_train.add(cids[0])

    candidate_val_clusters = [c for c in cluster_to_indices if c not in must_be_train]
    rng.shuffle(candidate_val_clusters)
    candidate_val_clusters.sort(key=lambda c: len(cluster_to_indices[c]))

    target_val_samples = int(len(train_indices) * val_ratio)
    inner_val_clusters = set()
    curr_val_samples = 0
    val_tool_counts = defaultdict(int)

    for cid in candidate_val_clusters:
        can_add = True
        for t in cluster_tools[cid]:
            if val_tool_counts[t] + 1 >= len(tool_to_cids[t]):
                can_add = False
                break
        if can_add:
            inner_val_clusters.add(cid)
            curr_val_samples += len(cluster_to_indices[cid])
            for t in cluster_tools[cid]:
                val_tool_counts[t] += 1
            if curr_val_samples >= target_val_samples:
                break

    inner_val_idx = []
    inner_train_idx = []
    for cid, idxs in cluster_to_indices.items():
        if cid in inner_val_clusters:
            inner_val_idx.extend(idxs)
        else:
            inner_train_idx.extend(idxs)

    inner_tr_tools = set(t for i in inner_train_idx for t in sample_tools[i])
    for t in all_tools:
        assert t in inner_tr_tools, f"Inner train missing tool {t}"

    return sorted(inner_train_idx), sorted(inner_val_idx)


def verify_coverage_and_leakage(
    outer_val_folds: List[List[int]],
    train_only_indices: List[int],
    cluster_ids: List[str],
    sample_tools: List[List[str]],
    raw_skeletons: List[str],
    all_tools: List[str],
) -> Dict[str, Any]:
    n_folds = len(outer_val_folds)
    audit = {}

    for f_idx in range(n_folds):
        val_idx = set(outer_val_folds[f_idx])
        train_idx = set(train_only_indices)
        for other_f in range(n_folds):
            if other_f != f_idx:
                train_idx.update(outer_val_folds[other_f])

        val_clusters = set(cluster_ids[i] for i in val_idx)
        train_clusters = set(cluster_ids[i] for i in train_idx)
        overlap = val_clusters.intersection(train_clusters)
        assert len(overlap) == 0, f"Fold {f_idx + 1} has cluster overlap: {overlap}"

        tr_tool_counts = defaultdict(int)
        tr_tool_clusters = defaultdict(set)
        for i in train_idx:
            cid = cluster_ids[i]
            for t in sample_tools[i]:
                tr_tool_counts[t] += 1
                tr_tool_clusters[t].add(cid)

        val_tool_counts = defaultdict(int)
        val_tool_clusters = defaultdict(set)
        for i in val_idx:
            cid = cluster_ids[i]
            for t in sample_tools[i]:
                val_tool_counts[t] += 1
                val_tool_clusters[t].add(cid)

        absent_tools = [t for t in all_tools if tr_tool_counts[t] == 0]
        assert len(absent_tools) == 0, f"Fold {f_idx + 1} has tools absent from training: {absent_tools}"
        assert tr_tool_counts["open_url"] > 0, "open_url missing from training"
        assert val_tool_counts["open_url"] == 0, "open_url must not appear in validation"

        # Near-duplicate check
        max_sim = 0.0
        val_skels_by_sig = defaultdict(set)
        for i in val_idx:
            sig, skel = raw_skeletons[i].split(" || ")
            val_skels_by_sig[sig].add(skel)

        tr_skels_by_sig = defaultdict(set)
        for i in train_idx:
            sig, skel = raw_skeletons[i].split(" || ")
            tr_skels_by_sig[sig].add(skel)

        for sig in val_skels_by_sig:
            if sig in tr_skels_by_sig:
                for v_skel in val_skels_by_sig[sig]:
                    for t_skel in tr_skels_by_sig[sig]:
                        sim = token_edit_sim(v_skel, t_skel)
                        if sim > max_sim:
                            max_sim = sim

        assert max_sim < 0.80, f"Fold {f_idx + 1} near-duplicate violation: max_sim = {max_sim:.4f} >= 0.80"

        audit[f"fold_{f_idx + 1}"] = {
            "train_sample_count": len(train_idx),
            "val_sample_count": len(val_idx),
            "train_cluster_count": len(train_clusters),
            "val_cluster_count": len(val_clusters),
            "max_token_edit_sim": round(max_sim, 4),
            "tools_absent_from_train": absent_tools,
            "tool_coverage": {
                t: {
                    "train_samples": tr_tool_counts[t],
                    "train_clusters": len(tr_tool_clusters[t]),
                    "val_samples": val_tool_counts[t],
                    "val_clusters": len(val_tool_clusters[t]),
                }
                for t in all_tools
            }
        }

    return audit


def create_all_folds_manifest():
    print("=== BUILDING TOOL-STRATIFIED GROUPED K-FOLD SPLITS (PHASE 1) ===", flush=True)
    repo_dir = Path(__file__).resolve().parent.parent
    raw_path = repo_dir / "data" / "raw" / "aether_orchestrator_dataset.jsonl"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_bytes = raw_path.read_bytes()
    dataset_sha256 = compute_sha256(raw_bytes)
    print(f"Dataset SHA-256 ({raw_path.name}): {dataset_sha256}", flush=True)

    frame_to_cluster = build_canonical_clusters(raw_path, sim_threshold=0.80)
    raw_skeletons = [extract_fine_semantic_frame(s) for s in raw_samples]
    cluster_ids = [frame_to_cluster[skel] for skel in raw_skeletons]

    dataset = ARNDataset(str(raw_path), tokenizer=TOKENIZER)
    sample_tools = []
    for item in dataset.encoded:
        active = [IDX_TO_TOOL[i] for i, val in enumerate(item["tool_labels"].tolist()) if val > 0.5]
        sample_tools.append(active)

    all_tools = sorted(list(set(t for tools in sample_tools for t in tools)))

    shuffles = [
        {"name": "shuffle_1", "seed": 42},
        {"name": "shuffle_2", "seed": 100},
        {"name": "shuffle_3", "seed": 2024},
    ]

    manifest = {
        "metadata": {
            "dataset_file": str(raw_path),
            "total_samples": len(raw_samples),
            "total_canonical_clusters": len(set(cluster_ids)),
            "sim_threshold": 0.80,
            "dataset_sha256": dataset_sha256,
            "train_only_tool": "open_url",
            "shuffles": shuffles,
            "n_folds_per_shuffle": 5,
            "total_outer_folds": 15,
        },
        "folds": {},
    }

    coverage_audit_new = {}

    for sh in shuffles:
        sh_name = sh["name"]
        seed = sh["seed"]
        print(f"\n--- Generating {sh_name} (seed={seed}) ---", flush=True)

        val_folds, train_only_indices = build_tool_stratified_folds(
            cluster_ids, sample_tools, raw_skeletons, n_folds=5, seed=seed
        )

        audit = verify_coverage_and_leakage(
            val_folds, train_only_indices, cluster_ids, sample_tools, raw_skeletons, all_tools
        )
        coverage_audit_new[sh_name] = audit

        sh_manifest = {}
        for f_idx in range(5):
            val_idx = sorted(val_folds[f_idx])
            train_idx = sorted(list(set(train_only_indices).union(
                *[val_folds[other_f] for other_f in range(5) if other_f != f_idx]
            )))

            inner_rng = random.Random(seed * 100 + f_idx)
            inner_train_idx, inner_val_idx = partition_inner_split(
                train_idx, cluster_ids, sample_tools, all_tools, inner_rng, val_ratio=0.20
            )

            fold_data = {
                "outer_train_indices": train_idx,
                "outer_val_indices": val_idx,
                "inner_train_indices": inner_train_idx,
                "inner_val_indices": inner_val_idx,
                "train_only_indices": sorted(train_only_indices),
                "metrics_summary": {
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "inner_train_samples": len(inner_train_idx),
                    "inner_val_samples": len(inner_val_idx),
                    "max_token_edit_sim": audit[f"fold_{f_idx + 1}"]["max_token_edit_sim"]
                }
            }

            fold_json_bytes = json.dumps(fold_data, sort_keys=True).encode("utf-8")
            fold_data["fold_sha256"] = compute_sha256(fold_json_bytes)
            sh_manifest[f"fold_{f_idx + 1}"] = fold_data

        manifest["folds"][sh_name] = sh_manifest

    manifest_path = repo_dir / "data" / "splits" / "cv_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved frozen cross-validation manifest to {manifest_path}", flush=True)

    audit_new_path = repo_dir / "results" / "coverage_audit_new_folds.json"
    audit_new_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_new_path, "w", encoding="utf-8") as f:
        json.dump(coverage_audit_new, f, indent=2)
    print(f"Saved new coverage audit to {audit_new_path}", flush=True)

    print("\n=== SUMMARY OF NEW TOOL-STRATIFIED FOLDS ===")
    print(f"{'Shuffle':12s} | {'Fold':8s} | {'Train N':8s} | {'Val N':8s} | {'Max Sim':8s} | {'Absent Tools':15s}")
    print("-" * 75)
    for sh_name, sh_audit in coverage_audit_new.items():
        for fold_name, f_data in sh_audit.items():
            print(f"{sh_name:12s} | {fold_name:8s} | {f_data['train_sample_count']:8d} | {f_data['val_sample_count']:8d} | {f_data['max_token_edit_sim']:8.4f} | None (100% covered)")


if __name__ == "__main__":
    create_all_folds_manifest()
