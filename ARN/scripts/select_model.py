"""
ARN Pre-registered Model Selection Script (Phase 7).
Reads ARN/results/ledger.json and applies the rules locked in docs/PREREGISTRATION.md.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


CANDIDATE_MODELS = [
    "frozen_arn",
    "frozen_cls_crf",
    "unfrozen_cls_crf",
    "unfrozen_arn",
    "unfrozen_jointbert",
]

SIMPLICITY_ORDER = {
    # Lower rank means simpler model (fewer trainable parameters / lower latency)
    "frozen_cls_crf": 1,
    "frozen_arn": 2,
    "unfrozen_cls_crf": 3,
    "unfrozen_arn": 4,
    "unfrozen_jointbert": 5,
}


def select_best_model(ledger_path: str = "ARN/results/ledger.json") -> Dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(f"Results ledger not found at {ledger_path}")

    with open(path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    # Aggregate candidate metrics across outer folds
    candidates_data = {}
    for entry in ledger:
        model = entry.get("model")
        protocol = entry.get("protocol")
        if model in CANDIDATE_MODELS and protocol == "phase4_master_cv":
            if model not in candidates_data:
                candidates_data[model] = {
                    "joint_em": [],
                    "tool_acc": [],
                    "span_f1": []
                }
            metrics = entry.get("metrics", {}).get("all_samples", {})
            candidates_data[model]["joint_em"].append(metrics.get("joint_em", 0.0))
            candidates_data[model]["tool_acc"].append(metrics.get("tool_subset_acc", 0.0))
            candidates_data[model]["span_f1"].append(metrics.get("strict_span_f1", 0.0))

    if not candidates_data:
        raise ValueError("No valid candidate evaluations found in ledger under protocol 'phase4_master_cv'.")

    # Compute means
    summary = {}
    for m, vals in candidates_data.items():
        summary[m] = {
            "mean_joint_em": sum(vals["joint_em"]) / len(vals["joint_em"]),
            "mean_tool_acc": sum(vals["tool_acc"]) / len(vals["tool_acc"]),
            "mean_span_f1": sum(vals["span_f1"]) / len(vals["span_f1"]),
            "n_folds": len(vals["joint_em"])
        }

    # Sort candidates by mean joint EM descending
    ranked_by_p2 = sorted(summary.items(), key=lambda x: x[1]["mean_joint_em"], reverse=True)
    top_model, top_stats = ranked_by_p2[0]

    # Check paired bootstrap differences from ledger comparisons
    comparisons = ledger_comparisons = {}
    for entry in ledger:
        if entry.get("protocol") == "phase4_paired_comparisons":
            comparisons = entry.get("comparisons", {})
            break

    # Simplicity rule check
    selected_model = top_model
    selection_reason = f"Top-performing model by Joint EM ({top_stats['mean_joint_em']:.4f})."

    for candidate, stats in ranked_by_p2[1:]:
        # Is candidate simpler?
        if SIMPLICITY_ORDER[candidate] < SIMPLICITY_ORDER[selected_model]:
            # Check if candidate is reliably worse than top_model
            pair_key = f"{selected_model}_vs_{candidate}"
            rev_pair_key = f"{candidate}_vs_{selected_model}"
            
            comp = comparisons.get(pair_key) or comparisons.get(rev_pair_key)
            if comp:
                delta = comp.get("mean_delta_joint_em", 0.0)
                ci = comp.get("ci_95_joint_em", [0.0, 0.0])
                ci_excludes_zero = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)
                
                # If top_model is not reliably better by >= 2.0% and CI excluding zero
                if not (ci_excludes_zero and abs(delta) >= 0.02):
                    selected_model = candidate
                    selection_reason = (
                        f"Simpler model {candidate} chosen over {top_model}; "
                        f"delta {abs(delta):.4f} is < 2.0% or 95% CI {ci} includes 0.0."
                    )
                    break

    # Determine Outcome A vs Outcome B
    # Check best unfrozen vs frozen ARN
    frozen_arn_stats = summary.get("frozen_arn", {})
    frozen_arn_p2 = frozen_arn_stats.get("mean_joint_em", 0.0)
    top_unfrozen = max(
        [(m, s) for m, s in summary.items() if m.startswith("unfrozen_")],
        key=lambda x: x[1]["mean_joint_em"]
    )
    unfrozen_delta = top_unfrozen[1]["mean_joint_em"] - frozen_arn_p2
    
    # Check unfrozen vs frozen ARN CI in comparisons
    arn_comp = comparisons.get(f"{top_unfrozen[0]}_vs_frozen_arn", {})
    arn_ci = arn_comp.get("ci_95_joint_em", [-1.0, 1.0])
    arn_ci_excludes_zero = (arn_ci[0] > 0 and arn_ci[1] > 0) or (arn_ci[0] < 0 and arn_ci[1] < 0)

    if unfrozen_delta >= 0.02 and arn_ci_excludes_zero:
        outcome = "Outcome A"
        outcome_desc = f"Fine-tuning superiority supported ({top_unfrozen[0]} beats frozen_arn by +{unfrozen_delta*100:.2f}%)."
    else:
        outcome = "Outcome B"
        outcome_desc = f"Frozen efficiency supported (delta +{unfrozen_delta*100:.2f}% is < 2.0% or CI {arn_ci} spans zero)."

    decision = {
        "ranked_candidates": summary,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "outcome": outcome,
        "outcome_description": outcome_desc
    }
    return decision


if __name__ == "__main__":
    try:
        dec = select_best_model()
        print(json.dumps(dec, indent=2))
    except Exception as e:
        print(f"select_model ready (pending ledger): {e}")
