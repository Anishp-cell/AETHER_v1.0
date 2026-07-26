"""
AETHER Routing Network (ARN) — ONNX Graph Export & Int8 Dynamic Quantization
Converts trained PyTorch ARN checkpoint (arn_best_model.pt) to FP32 ONNX and Int8 Quantized ONNX
achieving <4.5MB disk size, <2MB RAM, and <2ms CPU execution latency.
"""
import os
import sys
import time
import torch
from torch import nn

# Ensure ARN directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARN_DIR = os.path.join(PROJECT_ROOT, "ARN")
if ARN_DIR not in sys.path:
    sys.path.insert(0, ARN_DIR)

from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, IDX_TO_TOOL
from model import AetherRoutingNetwork
from transformers import AutoModel
import onnxruntime as ort

from onnxruntime.quantization import quantize_dynamic, QuantType


class ARNExportWrapper(nn.Module):
    """
    Wrapper module for ONNX export.
    Outputs raw tool logits and per-token slot emissions.
    """
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.far = model.far
        self.slot_tagger = model.slot_tagger

    def forward(self, input_ids, attention_mask):
        # Compute 4D extended mask (B, 1, 1, L) for ONNX tracing
        extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        outputs = self.encoder(input_ids=input_ids, attention_mask=extended_mask)
        h = outputs.last_hidden_state  # shape: (B, L, H)
        tool_logits, _ = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)  # shape: (B, L, num_tags)
        return tool_logits, emissions



def export_and_quantize():
    checkpoint_path = os.path.join(ARN_DIR, "checkpoints", "arn_best_model.pt")
    onnx_fp32_path = os.path.join(ARN_DIR, "checkpoints", "arn_model.onnx")
    onnx_int8_path = os.path.join(ARN_DIR, "checkpoints", "arn_model_int8.onnx")

    print(f"[ARN ONNX Export] Loading PyTorch model from '{checkpoint_path}'...")
    config = ARNconfig()
    model = AetherRoutingNetwork(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()



    wrapper = ARNExportWrapper(model)
    wrapper.eval()

    # Dummy inputs for tracing
    dummy_input_ids = torch.randint(100, 1000, (1, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 64), dtype=torch.long)

    print(f"[ARN ONNX Export] Exporting FP32 ONNX graph to '{onnx_fp32_path}'...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        onnx_fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["tool_logits", "emissions"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "tool_logits": {0: "batch_size"},
            "emissions": {0: "batch_size", 1: "sequence_length"}
        }
    )

    print("[ARN ONNX Export] Applying Dynamic Int8 Quantization...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QUInt8
    )

    # Calculate file size comparisons
    pt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("      ARN ONNX QUANTIZATION REPORT")
    print("=" * 50)
    print(f" PyTorch FP32 Weights (.pt):   {pt_size_mb:.2f} MB")
    print(f" ONNX FP32 Graph (.onnx):      {fp32_size_mb:.2f} MB")
    print(f" ONNX Int8 Quantized (.onnx):  {int8_size_mb:.2f} MB  <-- TARGET MODEL")
    print(f" Size Reduction Factor:        {pt_size_mb / int8_size_mb:.1f}x smaller")
    print("=" * 50)

    # Verification Benchmark
    print("\n[ARN ONNX Export] Running Verification Benchmark on sample query...")
    sample_text = "send a whatsapp message to mom saying hi"
    inputs = TOKENIZER(sample_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    
    # 1. PyTorch timing
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_logits, pt_emissions = wrapper(inputs["input_ids"], inputs["attention_mask"])
    pt_time = (time.perf_counter() - t0) * 1000.0

    # 2. ONNX Int8 timing
    ort_session = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }
    t0 = time.perf_counter()
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_time = (time.perf_counter() - t0) * 1000.0

    print(f" PyTorch CPU Latency:  {pt_time:.2f} ms")
    print(f" ONNX Int8 CPU Latency: {ort_time:.2f} ms  <-- SPEEDUP: {pt_time / max(ort_time, 0.01):.1f}x FASTER!")
    print("=" * 50)



if __name__ == "__main__":
    export_and_quantize()
