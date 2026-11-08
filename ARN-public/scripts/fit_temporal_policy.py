"""
Fit the temporal canonicalizer policy on the Mobile-Actions TRAIN split, then
report accuracy once on the eval split (oracle text, i.e. the full utterance).

Usage:
    python scripts/fit_temporal_policy.py --data <path/to/mobile-actions/dataset.jsonl> \
        [--out results/temporal_policy.json]

The eval split is read only after the policy is frozen; its number is a report,
never a selection criterion.
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Load the stdlib-only canonicalizer module directly so this script does not
# trigger arn/__init__.py (which imports torch/transformers and a tokenizer).
_TEMPORAL = Path(__file__).resolve().parents[1] / "src" / "arn" / "canon" / "temporal.py"
_spec = importlib.util.spec_from_file_location("arn_canon_temporal", _TEMPORAL)
temporal = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = temporal  # dataclasses resolve the module via sys.modules
_spec.loader.exec_module(temporal)
accuracy, fit_policy, iter_datetime_examples = (
    temporal.accuracy, temporal.fit_policy, temporal.iter_datetime_examples
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/temporal_policy.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    train = list(iter_datetime_examples(r for r in rows if r.get("metadata") == "train"))
    policy, train_acc = fit_policy(train)

    # Policy is frozen from here on.
    ev = list(iter_datetime_examples(r for r in rows if r.get("metadata") == "eval"))
    eval_acc = accuracy(ev, policy)

    report = {
        "policy": asdict(policy),
        "train_examples": len(train),
        "train_accuracy": round(train_acc, 4),
        "eval_examples": len(ev),
        "eval_accuracy_oracle_text": round(eval_acc, 4),
        "baseline_legacy_resolver_eval": 0.751,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
