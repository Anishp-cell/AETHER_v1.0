"""
ARN Checkpoint Expansion Utility
Surgically transfers an old 16-tool checkpoint into the new 18-tool model.

Usage (run from ARN/ directory with .venv active):
    python expand_checkpoint.py
    python expand_checkpoint.py --old checkpoints/arn_best_model.pt --out checkpoints/arn_expanded_18tool.pt

After this script finishes, use the output .pt as --checkpoint in train.py fine-tune.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

# ensure ARN is on sys.path
ARN_DIR = os.path.dirname(os.path.abspath(__file__))
if ARN_DIR not in sys.path:
    sys.path.insert(0, ARN_DIR)

from config import ARNconfig
from model import AetherRoutingNetwork


def expand_checkpoint(old_path: str, out_path: str):
    print(f"\n{'='*60}")
    print("ARN Checkpoint Expansion: 16-tool -> 18-tool")
    print(f"{'='*60}")
    print(f"  Input checkpoint : {old_path}")
    print(f"  Output checkpoint: {out_path}")

    if not os.path.exists(old_path):
        print(f"\n[ERROR] Old checkpoint not found: {old_path}")
        sys.exit(1)

    print("\n[Step 1] Loading old 16-tool checkpoint...")
    old_state = torch.load(old_path, map_location="cpu")
    print(f"  Loaded {len(old_state)} parameter tensors.")

    print("\n[Step 2] Instantiating new 18-tool AetherRoutingNetwork...")
    config = ARNconfig()
    model = AetherRoutingNetwork(config)
    new_state = model.state_dict()
    print(f"  New model has {len(new_state)} parameter tensors.")
    print(f"  NUM_TOOLS = {config.NUM_TOOLS}")

    print("\n[Step 3] Performing surgical weight transfer...")
    transferred_exact = 0
    transferred_surgical = 0
    skipped = 0

    for key in new_state:
        if key not in old_state:
            print(f"  [NEW   ] {key} -- keeping random init")
            skipped += 1
            continue

        old_t = old_state[key]
        new_t = new_state[key]

        if old_t.shape == new_t.shape:
            new_state[key] = old_t
            transferred_exact += 1
        elif key == "far.tool_embeddings.weight":
            # Tool count expanded 16 -> 18. Copy old 16 tools, keep new init for 16, 17
            new_t[0:16] = old_t[0:16]
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)} (transferred 16 tools)")
            transferred_surgical += 1
        elif key == "slot_tagger.weight":
            # Tag count expanded 22 -> 26.
            # Old 0..20: active tags, 21: <PAD>
            # New 0..20: active tags, 21..24: new tags, 25: <PAD>
            new_t[0:21] = old_t[0:21]
            new_t[25] = old_t[21]  # preserve <PAD> weights
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)} (remapped <PAD> 21->25)")
            transferred_surgical += 1
        elif key == "slot_tagger.bias":
            new_t[0:21] = old_t[0:21]
            new_t[25] = old_t[21]
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)} (remapped <PAD> 21->25)")
            transferred_surgical += 1
        elif key in ("crf.start_transitions", "crf.end_transitions"):
            new_t[0:21] = old_t[0:21]
            new_t[25] = old_t[21]
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)} (remapped <PAD> 21->25)")
            transferred_surgical += 1
        elif key == "crf.transitions":
            # 2D transitions matrix: [22, 22] -> [26, 26]
            new_t[0:21, 0:21] = old_t[0:21, 0:21]
            new_t[0:21, 25] = old_t[0:21, 21]
            new_t[25, 0:21] = old_t[21, 0:21]
            new_t[25, 25] = old_t[21, 21]
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)} (remapped <PAD> 21->25)")
            transferred_surgical += 1
        elif old_t.dim() == new_t.dim():
            # Generic fallback: copy overlapping slice
            slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_t.shape))
            new_t[slices] = old_t[slices]
            new_state[key] = new_t
            print(f"  [SURG  ] {key}: {list(old_t.shape)} -> {list(new_t.shape)}")
            transferred_surgical += 1
        else:
            print(f"  [SKIP  ] {key}: dim mismatch -- keeping random init")
            skipped += 1

    print(f"\n  Summary:")
    print(f"    Exact copies    : {transferred_exact}")
    print(f"    Surgical copies : {transferred_surgical}")
    print(f"    Skipped (new)   : {skipped}")

    print("\n[Step 4] Loading expanded state dict into model...")
    model.load_state_dict(new_state)
    model.eval()

    dummy_ids   = torch.randint(100, 1000, (1, 64), dtype=torch.long)
    dummy_mask  = torch.ones((1, 64), dtype=torch.long)
    with torch.no_grad():
        tool_preds, tag_preds, _ = model(dummy_ids, dummy_mask)
    assert tool_preds.shape == (1, config.NUM_TOOLS), \
        f"Expected tool_preds shape (1, {config.NUM_TOOLS}), got {tool_preds.shape}"
    print(f"  Sanity check passed: tool_preds.shape = {tuple(tool_preds.shape)}")

    print(f"\n[Step 5] Saving expanded checkpoint to: {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(new_state, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved! File size: {size_mb:.2f} MB")
    print(f"\n{'='*60}")
    print(f"Expansion complete. Now run fine-tune with:")
    print(f"  python train.py --checkpoint {out_path} --epochs 10 --lr 5e-5")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Expand ARN checkpoint from 16 to 18 tools.")
    parser.add_argument("--old", type=str,
                        default=os.path.join(ARN_DIR, "checkpoints", "arn_best_model.pt"),
                        help="Path to the original 16-tool checkpoint (.pt).")
    parser.add_argument("--out", type=str,
                        default=os.path.join(ARN_DIR, "checkpoints", "arn_expanded_18tool.pt"),
                        help="Output path for the expanded 18-tool checkpoint.")
    args = parser.parse_args()
    expand_checkpoint(args.old, args.out)


if __name__ == "__main__":
    main()