import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.dataset import ARNDataset
from arn.tokenizer import TOKENIZER, IDX_TO_TOOL
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters

raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
frame_to_cluster = build_canonical_clusters(raw_path)

dataset = ARNDataset(str(raw_path), tokenizer=TOKENIZER)
print(f"Dataset successfully parsed: {len(dataset.encoded)} samples")

with open(raw_path, "r", encoding="utf-8") as f:
    raw_samples = [json.loads(line) for line in f if line.strip()]

cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]
print(f"Total canonical clusters across 5,000 samples: {len(set(cluster_ids))}")

# Map active tools per sample
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

print("\n--- TOOL DISTRIBUTION ACROSS ALL 5,000 SAMPLES ---")
for t in sorted(tool_to_samples.keys()):
    print(f"  {t:30s}: {tool_to_samples[t]:4d} samples, {len(tool_to_clusters[t]):3d} clusters")
