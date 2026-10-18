"""
Inference and Prediction Pipeline for ARN.

Provides clean extraction of:
- Activated tool APIs with calibrated sigmoid confidence scores
- Argument spans mapped to slot entity names with character-offset alignment
- Decision confidence tracking and cloud LLM fallback flagging
- Deterministic, crash-free structured JSON output by design
"""

import re
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from transformers import BertTokenizerFast

from .config import ARNConfig
from .tokenizer import TOKENIZER, IDX_TO_TOOL, IDX_TO_TAG, TAG_TO_IDX


def extract_slot_arguments(
    user_text: str,
    tokenizer: BertTokenizerFast,
    tag_ids: List[int],
    max_seq_len: int = 64
) -> Dict[str, str]:
    """
    Extract argument slot entities directly from raw text using tokenizer offset mappings.
    Prevents subword artifact mangling (e.g., '##ing') by slicing raw character spans.
    """
    tokenized = tokenizer(
        user_text,
        return_offsets_mapping=True,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length"
    )
    offsets = tokenized["offset_mapping"]

    arguments: Dict[str, str] = {}
    current_slot: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    for i, tag_id in enumerate(tag_ids):
        tag = IDX_TO_TAG.get(tag_id, "O")
        start_char, end_char = offsets[i]

        # Ignore special tokens and padding
        if start_char == end_char or tag == "<PAD>":
            continue

        if tag.startswith("B-"):
            if current_slot is not None and span_start is not None and span_end is not None:
                val = user_text[span_start:span_end].strip("',.\"?;:")
                if val:
                    arguments[current_slot] = val

            current_slot = tag.split("-")[1]
            span_start = start_char
            span_end = end_char

        elif tag.startswith("I-"):
            tag_slot = tag.split("-")[1]
            if current_slot == tag_slot:
                span_end = end_char

        elif tag == "O":
            if current_slot is not None and span_start is not None and span_end is not None:
                val = user_text[span_start:span_end].strip("',.\"?;:")
                if val:
                    arguments[current_slot] = val
            current_slot = None
            span_start = None
            span_end = None

    if current_slot is not None and span_start is not None and span_end is not None:
        val = user_text[span_start:span_end].strip("',.\"?;:")
        if val:
            arguments[current_slot] = val

    return arguments


def predict(
    text: str,
    model: nn.Module,
    tokenizer: Optional[BertTokenizerFast] = None,
    config: Optional[ARNConfig] = None
) -> Dict[str, Any]:
    """
    End-to-end inference pass on a raw user utterance.

    Parameters
    ----------
    text : str
        User natural language command.
    model : nn.Module
        Trained AetherRoutingNetwork model.
    tokenizer : Optional[BertTokenizerFast]
        Tokenizer instance (defaults to standard BertTokenizerFast).
    config : Optional[ARNConfig]
        Configuration object (defaults to standard ARNConfig).

    Returns
    -------
    Dict[str, Any]
        Structured JSON dictionary containing:
        - "tools": List of activated tool names
        - "confidence": Float indicating top routing confidence (sigmoid probability)
        - "tool_confidences": Dict of {tool_name: float_probability} for active tools
        - "arguments": Dict of {slot_name: argument_value}
        - "cloud_fallback": Bool indicating whether confidence fell below cloud fallback threshold
        - "rejected": Bool indicating whether the input was classified as non-actionable (0 tools)
    """
    cfg = config if config is not None else ARNConfig()
    tok = tokenizer if tokenizer is not None else TOKENIZER
    device = next(model.parameters()).device

    model.eval()

    inputs = tok(
        text,
        max_length=cfg.max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        if hasattr(model, "tool_classifier"):
            tool_logits, emissions = model(input_ids, attention_mask)
            tool_probs = torch.sigmoid(tool_logits)
            tau = getattr(cfg, "optimal_threshold", 0.30)
            tool_preds = (tool_probs >= tau)
            tag_preds = model.crf.viterbi_decode(emissions, mask=attention_mask.bool())
        else:
            tool_probs, tool_preds, tag_preds, _ = model(input_ids, attention_mask)

    # Process tool probabilities and activations
    probs_row = tool_probs[0].cpu()
    preds_row = tool_preds[0].cpu()
    active_indices = torch.where(preds_row)[0].tolist()

    active_tools: List[str] = [IDX_TO_TOOL[idx] for idx in active_indices]
    tool_confidences: Dict[str, float] = {
        IDX_TO_TOOL[idx]: round(float(probs_row[idx].item()), 4)
        for idx in active_indices
    }

    # Top routing confidence is the maximum sigmoid probability across all tools
    max_conf = float(torch.max(probs_row).item()) if probs_row.numel() > 0 else 0.0

    # Extract argument slots
    tag_ids = tag_preds[0]
    arguments = extract_slot_arguments(
        user_text=text,
        tokenizer=tok,
        tag_ids=tag_ids,
        max_seq_len=cfg.max_seq_len
    )

    is_rejected = len(active_tools) == 0
    cloud_fallback = (not is_rejected) and (max_conf < cfg.cloud_fallback_threshold)

    return {
        "tools": active_tools,
        "confidence": round(max_conf, 4),
        "tool_confidences": tool_confidences,
        "arguments": arguments,
        "cloud_fallback": cloud_fallback,
        "rejected": is_rejected
    }
