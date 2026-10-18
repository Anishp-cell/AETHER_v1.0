"""
ONNX Export and Dynamic INT8 Quantization (Phase 8).

Exports the final CLSCRFModel to:
  1. FP32 ONNX graph: checkpoints/arn_final_model/model.onnx
  2. Dynamic INT8 quantized ONNX graph: checkpoints/arn_final_model/model_int8.onnx

Verifies numerical parity between PyTorch FP32 and ONNX FP32 runtime outputs (tol <= 1e-4).
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arn.baselines import CLSCRFModel
from arn.tokenizer import TOKENIZER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.export_onnx")


class ONNXExportWrapper(nn.Module):
    """Wraps CLSCRFModel encoder + heads for ONNX export (excluding non-differentiable CRF)."""
    def __init__(self, model: CLSCRFModel):
        super().__init__()
        self.encoder = model.encoder
        self.tool_classifier = model.tool_classifier
        self.slot_tagger = model.slot_tagger

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 4D extended attention mask: (B, 1, 1, seq_len)
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        outputs = self.encoder(input_ids=input_ids, attention_mask=ext_mask, return_dict=False)
        last_hidden = outputs[0]
        cls_hidden = last_hidden[:, 0, :]
        tool_logits = self.tool_classifier(cls_hidden)
        emissions = self.slot_tagger(last_hidden)
        return tool_logits, emissions


def export_and_quantize():
    repo_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = repo_dir / "checkpoints" / "arn_final_model"

    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.pt"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info("Loading PyTorch model weights...")
    model = CLSCRFModel(num_tools=config["num_tools"], num_tags=config["num_tags"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    wrapper = ONNXExportWrapper(model)
    wrapper.eval()

    # Dummy inputs for export (batch_size=1, seq_len=32)
    dummy_input_ids = torch.randint(100, 2000, (1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    # 1. Trace model to guarantee TorchScript graph stability
    logger.info("Tracing wrapper model...")
    traced_model = torch.jit.trace(wrapper, (dummy_input_ids, dummy_attention_mask))

    # 2. Export FP32 ONNX
    onnx_fp32_path = checkpoint_dir / "model.onnx"
    logger.info(f"Exporting FP32 ONNX graph to {onnx_fp32_path}...")

    torch.onnx.export(
        traced_model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["tool_logits", "emissions"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "tool_logits": {0: "batch_size"},
            "emissions": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"FP32 ONNX graph valid! Size: {onnx_fp32_path.stat().st_size / (1024*1024):.2f} MB")

    # 2. Verify parity between PyTorch and ONNX Runtime
    with torch.no_grad():
        pt_tool_logits, pt_emissions = wrapper(dummy_input_ids, dummy_attention_mask)

    ort_session = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }
    ort_tool_logits, ort_emissions = ort_session.run(None, ort_inputs)

    max_diff_tools = np.max(np.abs(pt_tool_logits.numpy() - ort_tool_logits))
    max_diff_slots = np.max(np.abs(pt_emissions.numpy() - ort_emissions))
    logger.info(f"Parity verification: max diff tools={max_diff_tools:.2e}, slots={max_diff_slots:.2e}")
    assert max_diff_tools < 1e-4, f"Tool logits mismatch: {max_diff_tools}"
    assert max_diff_slots < 1e-4, f"Slot emissions mismatch: {max_diff_slots}"
    logger.info("Numerical parity confirmed (tol < 1e-4)!")

    # 3. Dynamic INT8 Quantization
    onnx_int8_path = checkpoint_dir / "model_int8.onnx"
    logger.info(f"Quantizing to dynamic INT8: {onnx_int8_path}...")
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )

    int8_size = onnx_int8_path.stat().st_size / (1024 * 1024)
    fp32_size = onnx_fp32_path.stat().st_size / (1024 * 1024)
    logger.info(f"INT8 Model saved! FP32={fp32_size:.2f} MB, INT8={int8_size:.2f} MB (Compression ratio: {fp32_size/int8_size:.2f}x)")

    # Verify INT8 runnable
    ort_int8_session = ort.InferenceSession(str(onnx_int8_path), providers=["CPUExecutionProvider"])
    int8_out = ort_int8_session.run(None, ort_inputs)
    logger.info("INT8 ONNX model ran successfully in ONNX Runtime!")


if __name__ == "__main__":
    export_and_quantize()
