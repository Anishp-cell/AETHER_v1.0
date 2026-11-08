import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.tokenizer import TOKENIZER, IDX_TO_TOOL
from arn.dataset import ARNDataset
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
train_path = Path("ARN/data/splits/group_disjoint_train.jsonl")

with open(train_path, "r", encoding="utf-8") as f:
    raw_samples = [json.loads(l) for l in f if l.strip()]

dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
frame_to_cluster = build_canonical_clusters(raw_path)

tool_frames = defaultdict(Counter)
tool_clusters = defaultdict(set)

for item, s in zip(dataset.encoded, raw_samples):
    active = [IDX_TO_TOOL[i] for i, val in enumerate(item["tool_labels"].tolist()) if val > 0.5]
    frame = extract_fine_semantic_frame(s)
    cid = frame_to_cluster[frame]
    for t in active:
        tool_frames[t][frame] += 1
        tool_clusters[t].add(cid)

for t in sorted(tool_clusters.keys()):
    print(f"{t}: {len(tool_clusters[t])} clusters, {len(tool_frames[t])} frames, {sum(tool_frames[t].values())} samples")
    if len(tool_clusters[t]) <= 3:
        print(f"   Clusters: {tool_clusters[t]}")
        print(f"   Frames: {dict(tool_frames[t])}")
