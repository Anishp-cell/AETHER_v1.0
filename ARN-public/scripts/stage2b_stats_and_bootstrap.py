"""
Stage 2b Item 6: Statistical Analysis & Cluster Bootstrap (B=1,000).

Analyzes:
1. Exact k-of-N rates for all 8 models/baselines (per-fold and pooled across 15 folds).
2. Paired fold-by-fold differences vs ARN: Delta = Model - ARN.
3. Paired t-tests and Wilcoxon signed-rank tests vs ARN.
4. Cluster-level bootstrap (B=1,000) over 209 clusters for individual metrics and paired differences vs ARN.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2b_bootstrap")


def load_all_results():
    stage2_path = Path("ARN/results/stage2_comparative_results.json")
    triv_path = Path("ARN/results/stage2b_trivial_baselines.json")

    with open(stage2_path, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)

    with open(triv_path, "r", encoding="utf-8") as f:
        triv_data = json.load(f)

    all_models = {}
    for m in ["arn", "joint_bert", "cls_crf", "far_no_crf"]:
        all_models[m] = stage2_data[m]["raw_folds"]

    for b in ["most_frequent", "tfidf_logistic", "tfidf_knn", "bert_tiny_full_finetune"]:
        all_models[b] = triv_data[b]["raw_folds"]

    return all_models, stage2_data["arn"]["raw_folds"]


def cluster_bootstrap_analysis(
    all_models: Dict[str, List[Dict]],
    arn_folds: List[Dict],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict:
    rng = np.random.RandomState(seed)

    # 1. Build cluster-to-fold mapping from ARN folds
    # In each fold, we have outer_clusters list of length N (e.g. 785 or 786)
    # Total samples across 15 folds = 11,778
    cluster_sample_counts = defaultdict(int)
    for f in arn_folds:
        for cid in f["outer_clusters"]:
            cluster_sample_counts[cid] += 1

    unique_clusters = sorted(list(cluster_sample_counts.keys()))
    k_clusters = len(unique_clusters)
    logger.info(f"Loaded {k_clusters} unique clusters across 15 outer folds.")
    assert k_clusters == 209, f"Expected 209 clusters, got {k_clusters}"

    # Build per-cluster evaluations
    # For each fold f in 0..14, each sample j has a cluster cid
    # We assign fold-level metric to that sample (cluster-level grouped observation)
    cluster_records = defaultdict(lambda: defaultdict(list))
    for f_idx in range(15):
        cids = arn_folds[f_idx]["outer_clusters"]
        for m_name, f_list in all_models.items():
            f_data = f_list[f_idx]
            tool_acc = f_data["tool_subset_acc"]
            tool_f1 = f_data.get("tool_micro_f1", 0.0)
            span_f1 = f_data.get("span_f1", None)
            joint_em = f_data.get("joint_em", None)

            for cid in cids:
                cluster_records[cid][m_name].append({
                    "tool_acc": tool_acc,
                    "tool_f1": tool_f1,
                    "span_f1": span_f1,
                    "joint_em": joint_em,
                })

    # Resample 209 clusters B times
    bootstrap_samples = {m: {"tool_acc": [], "tool_f1": [], "span_f1": [], "joint_em": []} for m in all_models}
    bootstrap_diffs = {m: {"tool_acc": [], "tool_f1": [], "span_f1": [], "joint_em": []} for m in all_models if m != "arn"}

    for _ in range(n_boot):
        sampled_cids = rng.choice(unique_clusters, size=k_clusters, replace=True)

        means_this_boot = {}
        for m_name in all_models:
            t_acc_vals = []
            t_f1_vals = []
            s_f1_vals = []
            j_em_vals = []

            for cid in sampled_cids:
                for rec in cluster_records[cid][m_name]:
                    t_acc_vals.append(rec["tool_acc"])
                    t_f1_vals.append(rec["tool_f1"])
                    if rec["span_f1"] is not None:
                        s_f1_vals.append(rec["span_f1"])
                    if rec["joint_em"] is not None:
                        j_em_vals.append(rec["joint_em"])

            mean_t_acc = float(np.mean(t_acc_vals))
            mean_t_f1 = float(np.mean(t_f1_vals))
            mean_s_f1 = float(np.mean(s_f1_vals)) if s_f1_vals else None
            mean_j_em = float(np.mean(j_em_vals)) if j_em_vals else None

            bootstrap_samples[m_name]["tool_acc"].append(mean_t_acc)
            bootstrap_samples[m_name]["tool_f1"].append(mean_t_f1)
            if mean_s_f1 is not None:
                bootstrap_samples[m_name]["span_f1"].append(mean_s_f1)
            if mean_j_em is not None:
                bootstrap_samples[m_name]["joint_em"].append(mean_j_em)

            means_this_boot[m_name] = {
                "tool_acc": mean_t_acc,
                "tool_f1": mean_t_f1,
                "span_f1": mean_s_f1,
                "joint_em": mean_j_em,
            }

        arn_boot = means_this_boot["arn"]
        for m_name in bootstrap_diffs:
            bootstrap_diffs[m_name]["tool_acc"].append(means_this_boot[m_name]["tool_acc"] - arn_boot["tool_acc"])
            bootstrap_diffs[m_name]["tool_f1"].append(means_this_boot[m_name]["tool_f1"] - arn_boot["tool_f1"])
            if means_this_boot[m_name]["span_f1"] is not None and arn_boot["span_f1"] is not None:
                bootstrap_diffs[m_name]["span_f1"].append(means_this_boot[m_name]["span_f1"] - arn_boot["span_f1"])
            if means_this_boot[m_name]["joint_em"] is not None and arn_boot["joint_em"] is not None:
                bootstrap_diffs[m_name]["joint_em"].append(means_this_boot[m_name]["joint_em"] - arn_boot["joint_em"])

    # Compute 95% CIs [2.5%, 97.5%]
    ci_results = {}
    for m_name in all_models:
        ci_results[m_name] = {}
        for metric, vals in bootstrap_samples[m_name].items():
            if vals:
                ci_results[m_name][metric] = [
                    float(np.percentile(vals, 2.5)),
                    float(np.percentile(vals, 97.5)),
                ]

    diff_ci_results = {}
    for m_name in bootstrap_diffs:
        diff_ci_results[m_name] = {}
        for metric, vals in bootstrap_diffs[m_name].items():
            if vals:
                diff_ci_results[m_name][metric] = {
                    "mean_diff": float(np.mean(vals)),
                    "ci_95": [
                        float(np.percentile(vals, 2.5)),
                        float(np.percentile(vals, 97.5)),
                    ],
                }

    return ci_results, diff_ci_results


def run_statistical_suite():
    all_models, arn_folds = load_all_results()

    # 1. Exact k-of-N summary across 15 folds
    k_of_n_summary = {}
    for m_name, f_list in all_models.items():
        total_samples = sum(f["n_samples"] for f in f_list)
        total_tool_correct = sum(f["tool_subset_acc_count"] for f in f_list)
        total_gold_spans = sum(f.get("span_gold_total", 0) for f in f_list)
        total_correct_spans = sum(f.get("span_correct", 0) for f in f_list)
        total_joint_correct = sum(f.get("joint_em_count", 0) for f in f_list)

        per_fold_details = []
        for i, f in enumerate(f_list):
            fold_info = {
                "fold_idx": i + 1,
                "n_samples": f["n_samples"],
                "tool_acc_count": f["tool_subset_acc_count"],
                "tool_acc_rate": f["tool_subset_acc"],
                "tool_micro_f1": f.get("tool_micro_f1", 0.0),
            }
            if "span_correct" in f:
                fold_info["span_correct"] = f["span_correct"]
                fold_info["span_gold_total"] = f["span_gold_total"]
                fold_info["span_f1"] = f["span_f1"]
            if "joint_em_count" in f:
                fold_info["joint_em_count"] = f["joint_em_count"]
                fold_info["joint_em_rate"] = f["joint_em"]
            per_fold_details.append(fold_info)

        k_of_n_summary[m_name] = {
            "pooled_n_samples": total_samples,
            "pooled_tool_acc_count": total_tool_correct,
            "pooled_tool_acc_rate": total_tool_correct / total_samples,
            "pooled_span_correct": total_correct_spans,
            "pooled_span_gold_total": total_gold_spans,
            "pooled_joint_em_count": total_joint_correct,
            "pooled_joint_em_rate": total_joint_correct / total_samples if total_joint_correct > 0 else 0.0,
            "per_fold": per_fold_details,
        }

    # 2. Paired tests across 15 folds vs ARN
    arn_tool_accs = np.array([f["tool_subset_acc"] for f in all_models["arn"]])
    arn_tool_f1s = np.array([f["tool_micro_f1"] for f in all_models["arn"]])
    arn_span_f1s = np.array([f.get("span_f1", 0.0) for f in all_models["arn"]])
    arn_joint_ems = np.array([f.get("joint_em", 0.0) for f in all_models["arn"]])

    paired_tests = {}
    for m_name in all_models:
        if m_name == "arn":
            continue
        m_tool_accs = np.array([f["tool_subset_acc"] for f in all_models[m_name]])
        m_tool_f1s = np.array([f.get("tool_micro_f1", 0.0) for f in all_models[m_name]])

        diff_acc = m_tool_accs - arn_tool_accs
        diff_f1 = m_tool_f1s - arn_tool_f1s

        t_acc, p_acc = stats.ttest_rel(m_tool_accs, arn_tool_accs)
        try:
            w_acc, p_w_acc = stats.wilcoxon(diff_acc)
        except Exception:
            w_acc, p_w_acc = None, None

        t_f1, p_f1 = stats.ttest_rel(m_tool_f1s, arn_tool_f1s)

        record = {
            "tool_subset_acc": {
                "mean_delta": float(np.mean(diff_acc)),
                "sd_delta": float(np.std(diff_acc, ddof=1)),
                "paired_t_stat": float(t_acc),
                "paired_t_pvalue": float(p_acc),
                "wilcoxon_stat": float(w_acc) if w_acc is not None else None,
                "wilcoxon_pvalue": float(p_w_acc) if p_w_acc is not None else None,
            },
            "tool_micro_f1": {
                "mean_delta": float(np.mean(diff_f1)),
                "sd_delta": float(np.std(diff_f1, ddof=1)),
                "paired_t_stat": float(t_f1),
                "paired_t_pvalue": float(p_f1),
            }
        }

        if "span_f1" in all_models[m_name][0]:
            m_span_f1s = np.array([f["span_f1"] for f in all_models[m_name]])
            diff_span = m_span_f1s - arn_span_f1s
            t_span, p_span = stats.ttest_rel(m_span_f1s, arn_span_f1s)
            record["span_f1"] = {
                "mean_delta": float(np.mean(diff_span)),
                "sd_delta": float(np.std(diff_span, ddof=1)),
                "paired_t_stat": float(t_span),
                "paired_t_pvalue": float(p_span),
            }

        if "joint_em" in all_models[m_name][0]:
            m_joint_ems = np.array([f["joint_em"] for f in all_models[m_name]])
            diff_joint = m_joint_ems - arn_joint_ems
            t_joint, p_joint = stats.ttest_rel(m_joint_ems, arn_joint_ems)
            record["joint_em"] = {
                "mean_delta": float(np.mean(diff_joint)),
                "sd_delta": float(np.std(diff_joint, ddof=1)),
                "paired_t_stat": float(t_joint),
                "paired_t_pvalue": float(p_joint),
            }

        paired_tests[m_name] = record

    # 3. Cluster Bootstrap (B=1,000)
    logger.info("Running Cluster Bootstrap (B=1,000) across 209 clusters...")
    ci_results, diff_ci_results = cluster_bootstrap_analysis(all_models, arn_folds, n_boot=1000, seed=42)

    # 4. Save Final Report
    final_output = {
        "k_of_n_summary": k_of_n_summary,
        "paired_tests_15folds_vs_arn": paired_tests,
        "cluster_bootstrap_individual_95ci": ci_results,
        "cluster_bootstrap_paired_diffs_vs_arn_95ci": diff_ci_results,
    }

    out_path = Path("ARN/results/stage2b_stats_and_bootstrap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Saved complete statistical & bootstrap analysis to {out_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 115)
    print("STAGE 2b STATISTICAL COMPARISON & BOOTSTRAP (15 IDENTICAL FOLDS, 209 CLUSTERS)")
    print("=" * 115)
    header = f"{'Model/Baseline':26s} | {'Pooled Tool Acc (k/N)':23s} | {'Mean ± SD':16s} | {'95% Cluster CI':20s} | {'Delta vs ARN [95% CI]':22s}"
    print(header)
    print("-" * 115)

    arn_mean = np.mean([f["tool_subset_acc"] for f in all_models["arn"]])
    for m in all_models:
        k = k_of_n_summary[m]["pooled_tool_acc_count"]
        n = k_of_n_summary[m]["pooled_n_samples"]
        rate = k_of_n_summary[m]["pooled_tool_acc_rate"]
        sd = float(np.std([f["tool_subset_acc"] for f in all_models[m]], ddof=1))
        ci = ci_results[m]["tool_acc"]
        ci_str = f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]"

        if m == "arn":
            delta_str = "REFERENCE"
        else:
            d_info = diff_ci_results[m]["tool_acc"]
            d_mean = d_info["mean_diff"]
            d_ci = d_info["ci_95"]
            delta_str = f"{d_mean*100:+.2f}% [{d_ci[0]*100:+.2f}%, {d_ci[1]*100:+.2f}%]"

        print(f"{m:26s} | {k:5d} / {n:5d} ({rate*100:5.2f}%) | {rate*100:5.2f}% ± {sd*100:4.2f}% | {ci_str:20s} | {delta_str:22s}")
    print("=" * 115)


if __name__ == "__main__":
    run_statistical_suite()
