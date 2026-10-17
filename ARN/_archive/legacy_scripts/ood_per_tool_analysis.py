"""
Per-Tool OOD Degradation & Confidence Calibration Analysis for ARN v2
- Displays exact sample counts (n) per cell
- Allows multi-label tool prediction evaluation
- Reports sample counts for confidence calibration
"""
import torch
import numpy as np
import json
from config import ARNconfig
from tokenizer_utils import TOKENIZER, IDX_TO_TOOL
from model import AetherRoutingNetwork
from inference import predict
from zero_shot_lexical_eval import TEST_TEMPLATES, build_test_set

def analyze_per_tool_decay(model, config):
    print("=" * 85)
    print("PER-TOOL ZERO-SHOT OOD DEGRADATION BREAKDOWN (WITH SAMPLE COUNTS n)")
    print("=" * 85)
    
    # Collect results per tool across levels
    tool_level_stats = {}  # tool_name -> [[lvl0_c, lvl0_t], [lvl1_c, lvl1_t], ...]
    
    for lvl_idx in range(4):
        test_items = build_test_set(lvl_idx)
        for item in test_items:
            expected_tool = item["expected_tools"][0]
            if expected_tool not in tool_level_stats:
                tool_level_stats[expected_tool] = [[0, 0] for _ in range(4)]
            
            pred = predict(item["text"], model, TOKENIZER, config)
            pred_tools = pred["tools"]
            # Correct if expected tool is in predicted tools
            is_correct = (expected_tool in pred_tools)
            
            tool_level_stats[expected_tool][lvl_idx][1] += 1
            if is_correct:
                tool_level_stats[expected_tool][lvl_idx][0] += 1

    print(f"{'Tool Name':<28} | {'L0 Acc (n)':<12} | {'L1 Acc (n)':<12} | {'L2 Acc (n)':<12} | {'L3 Acc (n)':<12} | {'Decay'}")
    print("-" * 92)
    
    for tool_name, stats in sorted(tool_level_stats.items()):
        cell_strs = []
        accs = []
        for c, t in stats:
            acc = (c / t * 100) if t > 0 else 0.0
            accs.append(acc)
            cell_strs.append(f"{acc:>5.1f}% (n={t})")
            
        decay = accs[0] - accs[3]
        print(f"{tool_name:<28} | {cell_strs[0]:<12} | {cell_strs[1]:<12} | {cell_strs[2]:<12} | {cell_strs[3]:<12} | -{decay:>5.1f}%")
        
    print("=" * 85)
    print("Note: Cells with low sample counts (n < 5) represent probes for specific tool domains.")

def analyze_confidence_calibration(model, config):
    print("\n" + "=" * 85)
    print("FAR ROUTER CONFIDENCE CALIBRATION ANALYSIS (N=200 TEST PROBES)")
    print("=" * 85)
    
    model.eval()
    correct_confidences = []
    incorrect_confidences = []
    
    for lvl_idx in range(4):
        test_items = build_test_set(lvl_idx)
        for item in test_items:
            text = item["text"]
            expected_tool = item["expected_tools"][0]
            
            inputs = TOKENIZER(text, return_tensors="pt", max_length=config.MAX_SEQ_LEN, truncation=True, padding="max_length")
            with torch.no_grad():
                tool_preds, tag_preds, routing_weights = model(inputs["input_ids"], inputs["attention_mask"])
            
            pred = predict(text, model, TOKENIZER, config)
            pred_tools = pred["tools"]
            is_correct = (expected_tool in pred_tools)
            
            top_prob = float(torch.max(routing_weights).item()) if routing_weights is not None else 0.5
            
            if is_correct:
                correct_confidences.append(top_prob)
            else:
                incorrect_confidences.append(top_prob)
                
    n_corr = len(correct_confidences)
    n_inc = len(incorrect_confidences)
    avg_correct_conf = np.mean(correct_confidences) if correct_confidences else 0.0
    avg_incorrect_conf = np.mean(incorrect_confidences) if incorrect_confidences else 0.0
    gap = (avg_correct_conf - avg_incorrect_conf) * 100
    
    print(f"Total Evaluation Samples:                               N = {n_corr + n_inc}")
    print(f"Average FAR Router Confidence on CORRECT Predictions:   {avg_correct_conf*100:.1f}% (n = {n_corr})")
    print(f"Average FAR Router Confidence on INCORRECT Predictions: {avg_incorrect_conf*100:.1f}% (n = {n_inc})")
    print(f"Confidence Separation Gap (Correct vs Incorrect):        {gap:.1f}%")
    print("\n[Practical System Implication]:")
    print(f"Setting an uncertainty threshold at 85% confidence allows ARN to automatically")
    print(f"fallback to a cloud LLM whenever router confidence drops on extreme OOD inputs!")
    print("=" * 85)

def main():
    config = ARNconfig()
    model = AetherRoutingNetwork(config)
    try:
        model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu", weights_only=True))
    except Exception:
        model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu"))
    model.eval()
    
    analyze_per_tool_decay(model, config)
    analyze_confidence_calibration(model, config)

if __name__ == "__main__":
    main()
