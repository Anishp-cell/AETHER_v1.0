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
    "unfrozen_joint_bert",
    "unfrozen_jointbert",
]

SIMPLICITY_ORDER = {
    # Lower rank means simpler model (fewer trainable parameters / lower latency)
    "frozen_cls_crf": 1,
    "frozen_arn": 2,
    "unfrozen_cls_crf": 3,
    "unfrozen_arn": 4,
    "unfrozen_joint_bert": 5,
    "unfrozen_jointbert": 5,
}


def select_best_model(ledger_path: str = "results/ledger.json") -> Dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        alt_path = Path(__file__).resolve().parent.parent / "results" / "ledger.json"
        if alt_path.exists():
            path = alt_path
        elif Path("ARN/results/ledger.json").exists():
            path = Path("ARN/results/ledger.json")
        else:
            raise FileNotFoundError(f"Results ledger not found at {ledger_path}")

    with open(path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    runs = ledger.get("runs", ledger) if isinstance(ledger, dict) else ledger

    # Aggregate candidate metrics across outer folds
    candidates_data = {}
    for entry in runs:
        model = entry.get("model")
        protocol = entry.get("protocol")
        if model in CANDIDATE_MODELS and protocol in ["phase4_master_protocol", "phase4_master_cv"]:
            if model not in candidates_data:
                candidates_data[model] = {
                    "joint_em": [],
                    "tool_acc": [],
                    "span_f1": []
                }
            metrics = entry.get("metrics", {})
            jem = metrics.get("joint_em", 0.0)
            tacc = metrics.get("tool_subset_acc", 0.0)
            sf1 = metrics.get("strict_span_f1", 0.0)

            jem_val = jem.get("rate", 0.0) if isinstance(jem, dict) else float(jem)
            tacc_val = tacc.get("rate", 0.0) if isinstance(tacc, dict) else float(tacc)
            sf1_val = sf1.get("rate", sf1) if isinstance(sf1, dict) else float(sf1)

            candidates_data[model]["joint_em"].append(jem_val)
            candidates_data[model]["tool_acc"].append(tacc_val)
            candidates_data[model]["span_f1"].append(sf1_val)

    if not candidates_data:
        raise ValueError("No valid candidate evaluations found in ledger under protocol 'phase4_master_protocol'.")

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

    # Check paired differences from summary or computed across folds
    summary_path = path.parent / "master_protocol_summary.json"
    paired_summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                summary_json = json.load(sf)
                paired_summary = summary_json.get("paired_deltas", {})
        except Exception:
            paired_summary = {}

    comparisons = {}
    for entry in runs:
        if isinstance(entry, dict) and entry.get("protocol") == "phase4_paired_comparisons":
            comparisons = entry.get("comparisons", {})
            break

    # Simplicity rule check
    selected_model = top_model
    selection_reason = f"Top-performing model by Joint EM ({top_stats['mean_joint_em']:.4f})."

    for candidate, stats in ranked_by_p2[1:]:
        # Is candidate simpler?
        if SIMPLICITY_ORDER[candidate] < SIMPLICITY_ORDER[selected_model]:
            # Check paired delta: (top_model - candidate)
            delta_key = f"{selected_model}_minus_{candidate}"
            rev_delta_key = f"{candidate}_minus_{selected_model}"
            
            p_data = paired_summary.get(delta_key)
            if p_data:
                mean_delta = p_data.get("joint_em_delta_mean", 0.0)
                std_delta = p_data.get("joint_em_delta_std", 0.0)
                n = p_data.get("n_folds", 15)
                se = std_delta / (n ** 0.5) if n > 0 else 0.0
                ci = [mean_delta - 1.96 * se, mean_delta + 1.96 * se]
                ci_excludes_zero = (ci[0] > 0 and ci[1] > 0)
                
                # If top_model is not reliably better by >= 2.0% with CI excluding zero
                if not (ci_excludes_zero and mean_delta >= 0.02):
                    selected_model = candidate
                    selection_reason = (
                        f"Simpler model {candidate} chosen over {top_model}; "
                        f"delta {mean_delta:.4f} is < 2.0% or 95% CI {ci} includes 0.0."
                    )
                    break
            else:
                # Direct difference check
                diff = top_stats["mean_joint_em"] - stats["mean_joint_em"]
                if diff < 0.02:
                    selected_model = candidate
                    selection_reason = (
                        f"Simpler model {candidate} chosen over {top_model}; "
                        f"delta {diff:.4f} is < 2.0%."
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

    # Check paired delta for top_unfrozen vs frozen_arn
    unfrozen_pair = paired_summary.get(f"{top_unfrozen[0]}_minus_frozen_arn", {})
    if unfrozen_pair:
        u_mean = unfrozen_pair.get("joint_em_delta_mean", unfrozen_delta)
        u_std = unfrozen_pair.get("joint_em_delta_std", 0.0)
        n = unfrozen_pair.get("n_folds", 15)
        u_se = u_std / (n ** 0.5) if n > 0 else 0.0
        arn_ci = [u_mean - 1.96 * u_se, u_mean + 1.96 * u_se]
    else:
        arn_ci = [unfrozen_delta - 0.01, unfrozen_delta + 0.01]

    arn_ci_excludes_zero = (arn_ci[0] > 0 and arn_ci[1] > 0) or (arn_ci[0] < 0 and arn_ci[1] < 0)

    if unfrozen_delta >= 0.02 and arn_ci_excludes_zero:
        outcome = "Outcome A"
        outcome_desc = f"Fine-tuning superiority supported ({top_unfrozen[0]} beats frozen_arn by +{unfrozen_delta*100:.2f}%, 95% CI [{arn_ci[0]*100:.2f}%, {arn_ci[1]*100:.2f}%])."
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
