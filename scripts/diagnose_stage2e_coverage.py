import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2c_head_independence import partition_clusters

train_path = Path("ARN/data/splits/group_disjoint_train.jsonl")
raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")

from arn.tokenizer import TOKENIZER, IDX_TO_TOOL

with open(train_path, "r", encoding="utf-8") as f:
    raw_samples = [json.loads(line) for line in f if line.strip()]

frame_to_cluster = build_canonical_clusters(raw_path)
cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
dataset = ARNDataset(train_path, tokenizer=TOKENIZER)

print(f"Total samples: {len(raw_samples)}")
print(f"Total unique clusters in train: {len(set(cluster_ids))}")

# Extract active tools per sample
sample_tools = []
for item in dataset.encoded:
    active = [IDX_TO_TOOL[i] for i, val in enumerate(item["tool_labels"].tolist()) if val > 0.5]
    sample_tools.append(active)

tool_to_clusters = defaultdict(set)
tool_to_samples = defaultdict(int)
for tools, cid in zip(sample_tools, cluster_ids):
    for t in tools:
        tool_to_clusters[t].add(cid)
        tool_to_samples[t] += 1

print("\n--- TOOL CLUSTER & SAMPLE COUNTS ---")
for t in sorted(tool_to_samples.keys()):
    print(f"{t}: {tool_to_samples[t]} samples, {len(tool_to_clusters[t])} clusters")

# Old folds inspection (seed 42)
old_folds = partition_clusters(cluster_ids, n_folds=5, seed=42)

print("\n--- OLD FOLD DESIGN (seed 42) COVERAGE AUDIT ---")
for f_idx in range(5):
    val_idx = set(old_folds[f_idx])
    train_idx = set(i for other_f in range(5) if other_f != f_idx for i in old_folds[other_f])
    
    val_tools = defaultdict(int)
    tr_tools = defaultdict(int)
    val_cl = defaultdict(set)
    tr_cl = defaultdict(set)
    
    for i in train_idx:
        for t in sample_tools[i]:
            tr_tools[t] += 1
            tr_cl[t].add(cluster_ids[i])
            
    for i in val_idx:
        for t in sample_tools[i]:
            val_tools[t] += 1
            val_cl[t].add(cluster_ids[i])
            
    absent_from_tr = [t for t in val_tools if tr_tools[t] == 0]
    unseen_val_samples = sum(1 for i in val_idx if any(tr_tools[t] == 0 for t in sample_tools[i]))
    share_unseen = unseen_val_samples / len(val_idx) * 100
    
    print(f"\nFold {f_idx + 1}: Train N={len(train_idx)}, Val N={len(val_idx)}")
    print(f"  Tools absent from training: {absent_from_tr}")
    print(f"  Val samples with unseen tool: {unseen_val_samples}/{len(val_idx)} ({share_unseen:.2f}%)")
    for t in sorted(set(list(tr_tools.keys()) + list(val_tools.keys()))):
        if tr_tools[t] == 0:
            print(f"    [ABSENT] {t}: Tr={tr_tools[t]} (0 cl) | Val={val_tools[t]} ({len(val_cl[t])} cl)")
