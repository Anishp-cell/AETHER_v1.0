#!/usr/bin/env python3
"""
Automated LaTeX Table Generator for ARN Manuscript.
Generates camera-ready LaTeX table snippets directly from experimental result JSON files.
Strictly eliminates manual data entry to ensure 100% reproducibility.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.make_tables")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning("Result file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_main_table(results_dir: Path, output_file: Path):
    """Generate Table 1: In-Distribution Joint Benchmark (ARN vs Baselines)."""
    models = ["arn", "jointbert", "cls_crf", "far_no_crf"]
    names = {
        "arn": "ARN (Ours)",
        "jointbert": "JointBERT (Chen et al. 2019)",
        "cls_crf": "CLS + CRF (Ablation: No FAR)",
        "far_no_crf": "FAR w/o CRF (Ablation: No CRF)",
    }

    rows = []
    for m in models:
        data = load_json(results_dir / f"{m}_eval.json")
        if data:
            tool_f1 = f"{100.0 * data['tool_routing']['macro_f1']:.1f}"
            slot_f1 = f"{100.0 * data['slot_tagging']['span_f1']:.1f}"
            joint_acc = f"{100.0 * data['joint_frame_accuracy']:.1f}"
            ece = f"{data['tool_routing']['calibration_ece']:.4f}"
        else:
            tool_f1, slot_f1, joint_acc, ece = "--", "--", "--", "--"
        rows.append(f"{names[m]} & {tool_f1} & {slot_f1} & {joint_acc} & {ece} \\\\")

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{In-distribution evaluation on group-disjoint test split ($K=18$ tools, 26 slot tags).}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\textbf{Architecture} & \\textbf{Tool Macro F1 (\\%)} & \\textbf{Slot Span F1 (\\%)} & \\textbf{Joint Frame Acc (\\%)} & \\textbf{Tool ECE} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info("Generated %s", output_file)


def generate_hardware_table(results_dir: Path, output_file: Path):
    """Generate Table 2: Hardware Latency and Memory on Host CPU."""
    fp32_lat = load_json(results_dir / "latency_fp32.json")
    int8_lat = load_json(results_dir / "latency_int8.json")
    fp32_mem = load_json(results_dir / "memory_fp32.json")
    int8_mem = load_json(results_dir / "memory_int8.json")

    def get_lat(d):
        if not d:
            return ("--", "--", "--")
        b = d.get("benchmark", {})
        return (f"{b.get('p50_ms', 0):.2f}", f"{b.get('p95_ms', 0):.2f}", f"{b.get('mean_ms', 0):.2f}")

    def get_mem(d):
        if not d:
            return ("--", "--", "--")
        return (f"{d.get('file_size_mb', 0):.2f}", f"{d.get('peak_rss_mb', 0):.2f}", f"{d.get('incremental_rss_mb', 0):.2f}")

    p50_f, p95_f, mean_f = get_lat(fp32_lat)
    disk_f, peak_f, inc_f = get_mem(fp32_mem)

    p50_i, p95_i, mean_i = get_lat(int8_lat)
    disk_i, peak_i, inc_i = get_mem(int8_mem)

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Runtime footprint and latency on laptop-class x86 CPU (Intel Core i7-13700HX, Windows 11, ONNX Runtime 1.23.2, 1 thread).}\n"
        "\\label{tab:hardware_bench}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "\\textbf{Precision} & \\textbf{Disk (MB)} & \\textbf{p50 (ms)} & \\textbf{p95 (ms)} & \\textbf{Mean (ms)} & \\textbf{Peak RSS (MB)} & \\textbf{Inc. RSS (MB)} \\\\\n"
        "\\midrule\n"
        f"ONNX FP32 & {disk_f} & {p50_f} & {p95_f} & {mean_f} & {peak_f} & {inc_f} \\\\\n"
        f"ONNX INT8 & {disk_i} & {p50_i} & {p95_i} & {mean_i} & {peak_i} & {inc_i} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info("Generated %s", output_file)


def main():
    parser = argparse.ArgumentParser(description="ARN Automated LaTeX Table Generator")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing JSON results")
    parser.add_argument("--output-dir", type=str, default="docs/paper/tables", help="Directory to save LaTeX tables")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_main_table(res_dir, out_dir / "table1_main_results.tex")
    generate_hardware_table(res_dir, out_dir / "table2_hardware.tex")


if __name__ == "__main__":
    main()
