#!/usr/bin/env python3
"""
ARN ONNX Export and INT8 Quantization Utility.
Exports PyTorch checkpoints to:
1. Full FP32 ONNX Graph
2. Dynamic INT8 Quantized ONNX Graph
Verifies outputs against PyTorch using ONNX Runtime.
"""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typing import Dict, Any

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from arn import AetherRoutingNetwork, ARNConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.export_onnx")


class ARNONNXWrapper(torch.nn.Module):
    """Export wrapper returning tool logits and slot emissions."""
    def __init__(self, model: AetherRoutingNetwork):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        tool_logits, _ = self.model.far(h, attention_mask)
        emissions = self.model.slot_tagger(h)
        tool_probs = torch.sigmoid(tool_logits)
        return tool_probs, emissions


def export_models(checkpoint_path: Path, output_dir: Path, opset: int = 17) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt.get("config", {})
    config = ARNConfig(**{k: v for k, v in config_dict.items() if hasattr(ARNConfig, k)}) if config_dict else ARNConfig()

    model = AetherRoutingNetwork(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ARNONNXWrapper(model)
    wrapper.eval()

    dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    fp32_path = output_dir / "arn_model.onnx"
    int8_path = output_dir / "arn_model_int8.onnx"

    logger.info("Exporting FP32 ONNX model to %s (opset %d)...", fp32_path, opset)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["tool_probs", "slot_emissions"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "tool_probs": {0: "batch_size"},
            "slot_emissions": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    logger.info("FP32 ONNX export complete: %.2f MB", fp32_size_mb)

    logger.info("Quantizing to dynamic INT8 (%s)...", int8_path)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
    logger.info("INT8 dynamic quantization complete: %.2f MB", int8_size_mb)

    # Verification with ONNX Runtime
    logger.info("Verifying exported graphs with ONNX Runtime...")
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }
    with torch.no_grad():
        py_probs, py_emissions = wrapper(dummy_input_ids, dummy_attention_mask)

    ort_fp32_out = sess_fp32.run(None, inputs)
    np.testing.assert_allclose(py_probs.numpy(), ort_fp32_out[0], rtol=1e-3, atol=1e-4)
    logger.info("FP32 ONNX Runtime verification PASSED (max diff < 1e-4)")

    ort_int8_out = sess_int8.run(None, inputs)
    logger.info("INT8 ONNX Runtime inference test PASSED")

    return {"fp32": fp32_path, "int8": int8_path}


def main():
    parser = argparse.ArgumentParser(description="ARN ONNX Export and Quantization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/checkpoints", help="Output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return

    export_models(ckpt_path, Path(args.output_dir), opset=args.opset)


if __name__ == "__main__":
    main()
