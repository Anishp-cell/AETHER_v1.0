"""
Evaluates trained FP32 vs INT8 ONNX accuracy on Fold 1 outer validation.
Trains Unfrozen CLS+CRF and Unfrozen ARN on Fold 1 train (10 epochs, GPU),
exports to ONNX FP32 and INT8, and measures accuracy degradation.
Updates ARN/results/stage2d_onnx_deployment.json.
"""

import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arn.config import ARNConfig
from arn.dataset import ARNDataset
from arn.baselines import CLSCRFModel
from arn.model import AetherRoutingNetwork
from arn.crf import CRF
from arn.far import FactoredAttentionRouter
from arn.tokenizer import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG
from scripts.make_splits import extract_fine_semantic_frame, build_canonical_clusters
from scripts.stage2c_head_independence import partition_clusters
from scripts.stage2c_softmax_baselines import extract_spans_strict, compute_span_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arn.stage2d_onnx_acc")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainedUnfrozenCLS_CRF(nn.Module):
    def __init__(self, num_tools=18, num_tags=26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.tool_classifier = nn.Linear(128, num_tools)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls_repr = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_repr)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


class TrainedUnfrozenARN(nn.Module):
    def __init__(self, num_tools=18, num_tags=26):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager")
        self.far = FactoredAttentionRouter(num_tools=num_tools, hidden_dim=128)
        self.slot_tagger = nn.Linear(128, num_tags)
        self.crf = CRF(num_tags=num_tags, hard_bio_mask=True, tag_to_idx=TAG_TO_IDX)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        tool_logits, _ = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        return tool_logits, emissions


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = model.encoder
        self.tool_classifier = getattr(model, "tool_classifier", None)
        self.far = getattr(model, "far", None)
        self.slot_tagger = getattr(model, "slot_tagger", None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        emb = self.encoder.embeddings(input_ids=input_ids)
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        enc_out = self.encoder.encoder(emb, attention_mask=ext_mask)
        h = enc_out[0]
        if self.tool_classifier is not None:
            tool_logits = self.tool_classifier(h[:, 0, :])
        else:
            tool_logits, _ = self.far(h, attention_mask)
        slot_emissions = self.slot_tagger(h)
        return tool_logits, slot_emissions


def collate_fn(batch, device):
    max_l = max(b["attention_mask"].sum().item() for b in batch)
    max_l = min(max_l, 64)
    ids = torch.stack([b["input_ids"][:max_l] for b in batch]).to(device)
    mask = torch.stack([b["attention_mask"][:max_l] for b in batch]).to(device)
    tags = torch.stack([b["tags"][:max_l] for b in batch]).to(device)
    tools = torch.stack([b["tool_labels"] for b in batch]).to(device)
    return ids, mask, tags, tools


def train_model(model: nn.Module, tr_data: List[Dict], device: torch.device, epochs=10):
    n_tr = len(tr_data)
    trunk_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": 3e-5, "weight_decay": 0.01},
        {"params": other_params, "lr": 3e-4, "weight_decay": 0.01},
    ])

    for ep in range(1, epochs + 1):
        model.train()
        perm = list(range(n_tr))
        random.shuffle(perm)
        for b in range(0, n_tr, 64):
            batch_idx = [perm[k] for k in range(b, min(b + 64, n_tr))]
            batch = [tr_data[k] for k in batch_idx]
            ids, mask, tags, tools = collate_fn(batch, device)

            opt.zero_grad()
            t_logits, emissions = model(ids, mask)
            loss_tool = model.bce(t_logits, tools.float())
            active_tokens = mask.sum(dim=1).clamp(min=1)
            crf_loss = model.crf.sentence_nll(emissions, tags, mask)
            loss_slot = (crf_loss / active_tokens).mean()
            total_loss = loss_tool + loss_slot
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def evaluate_ort_session(ort_session: ort.InferenceSession, crf_module: CRF, vl_data: List[Dict], tau: float = 0.50):
    all_tool_probs = []
    gold_tools = []
    all_pred_spans = []
    gold_spans = []

    for item in vl_data:
        in_ids = item["input_ids"].unsqueeze(0).numpy().astype(np.int64)
        att_m = item["attention_mask"].unsqueeze(0).numpy().astype(np.int64)
        t_logits, s_emissions = ort_session.run(None, {"input_ids": in_ids, "attention_mask": att_m})

        probs = 1.0 / (1.0 + np.exp(-t_logits[0]))
        all_tool_probs.append(probs)
        gold_tools.append(item["tool_labels"].numpy())

        l_i = int(att_m[0].sum())
        em_tensor = torch.from_numpy(s_emissions)
        m_tensor = torch.from_numpy(att_m)
        pred_tags = crf_module.viterbi_decode(em_tensor, m_tensor)[0]
        p_seq = [IDX_TO_TAG.get(idx, "O") for idx in pred_tags[:l_i]]
        all_pred_spans.append(extract_spans_strict(p_seq))

        g_seq = [IDX_TO_TAG.get(idx, "O") for idx in item["tags"][:l_i].tolist()]
        gold_spans.append(extract_spans_strict(g_seq))

    all_tool_probs = np.vstack(all_tool_probs)
    gold_tools = np.vstack(gold_tools)
    pred_tools = (all_tool_probs >= tau).astype(int)

    tool_matches = np.all(pred_tools == gold_tools, axis=1)
    tool_acc = float(np.mean(tool_matches))

    span_metrics = compute_span_metrics(gold_spans, all_pred_spans)
    slot_matches = [g == p for g, p in zip(gold_spans, all_pred_spans)]
    joint_em = float(np.mean([t and s for t, s in zip(tool_matches, slot_matches)]))

    return {
        "tool_subset_acc": tool_acc,
        "span_f1": span_metrics["strict_f1"],
        "joint_em": joint_em,
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING & BENCHMARKING ONNX FP32 vs INT8 ACCURACY ON {device} ===", flush=True)

    train_path = "ARN/data/splits/group_disjoint_train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    raw_path = Path("ARN/data/raw/aether_orchestrator_dataset.jsonl")
    frame_to_cluster = build_canonical_clusters(raw_path)
    dataset = ARNDataset(train_path, tokenizer=TOKENIZER)
    cluster_ids = [frame_to_cluster[extract_fine_semantic_frame(s)] for s in raw_samples]

    outer_folds = partition_clusters(cluster_ids, n_folds=5, seed=42)
    val_idx = outer_folds[0]
    train_idx = [i for other_f in range(1, 5) for i in outer_folds[other_f]]

    tr_data = [dataset.encoded[i] for i in train_idx]
    vl_data = [dataset.encoded[i] for i in val_idx]
    print(f"Fold 1: Train N={len(tr_data)}, Val N={len(vl_data)}", flush=True)

    export_dir = Path("ARN/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("unfrozen_cls_crf", TrainedUnfrozenCLS_CRF()),
        ("unfrozen_arn", TrainedUnfrozenARN()),
    ]

    dummy_input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 64), dtype=torch.long)

    # Load existing stage2d_onnx_deployment.json
    deploy_file = Path("ARN/results/stage2d_onnx_deployment.json")
    with open(deploy_file, "r", encoding="utf-8") as f:
        deploy_data = json.load(f)

    for model_name, model in models:
        print(f"\n--- Training {model_name} (10 epochs) ---", flush=True)
        model.to(device)
        train_model(model, tr_data, device, epochs=10)
        model.eval()
        model.to("cpu")

        crf_module = model.crf

        # Export FP32 ONNX
        wrapper = ExportWrapper(model)
        fp32_path = export_dir / f"{model_name}_fp32.onnx"
        int8_path = export_dir / f"{model_name}_int8.onnx"

        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["tool_logits", "slot_emissions"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "tool_logits": {0: "batch_size"},
                "slot_emissions": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )

        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        # Evaluate FP32 ONNX
        sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        metrics_fp32 = evaluate_ort_session(sess_fp32, crf_module, vl_data, tau=0.50)
        print(f"  FP32 ONNX: Tool Acc={metrics_fp32['tool_subset_acc']*100:.2f}%, Span F1={metrics_fp32['span_f1']*100:.2f}%, Joint EM={metrics_fp32['joint_em']*100:.2f}%", flush=True)

        # Evaluate INT8 ONNX
        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        metrics_int8 = evaluate_ort_session(sess_int8, crf_module, vl_data, tau=0.50)
        print(f"  INT8 ONNX: Tool Acc={metrics_int8['tool_subset_acc']*100:.2f}%, Span F1={metrics_int8['span_f1']*100:.2f}%, Joint EM={metrics_int8['joint_em']*100:.2f}%", flush=True)

        delta_tool = (metrics_int8["tool_subset_acc"] - metrics_fp32["tool_subset_acc"]) * 100
        delta_span = (metrics_int8["span_f1"] - metrics_fp32["span_f1"]) * 100
        delta_joint = (metrics_int8["joint_em"] - metrics_fp32["joint_em"]) * 100
        print(f"  Quantization Delta (INT8 - FP32): Tool={delta_tool:+.2f}%, Span F1={delta_span:+.2f}%, Joint EM={delta_joint:+.2f}%", flush=True)

        # Update json
        deploy_data["models"][model_name]["fp32_validation_accuracy"] = metrics_fp32
        deploy_data["models"][model_name]["int8_validation_accuracy"] = metrics_int8
        deploy_data["models"][model_name]["quantization_deltas"] = {
            "tool_subset_acc_delta": delta_tool,
            "span_f1_delta": delta_span,
            "joint_em_delta": delta_joint,
        }

    with open(deploy_file, "w", encoding="utf-8") as f:
        json.dump(deploy_data, f, indent=2)
    print(f"\nUpdated {deploy_file} successfully!", flush=True)


if __name__ == "__main__":
    main()
