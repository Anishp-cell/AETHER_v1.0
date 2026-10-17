"""
Dataset and Data Loading Utilities for ARN.

Loads JSONL datasets formatted with OpenAI-style message schemas,
maps tool calls into multi-label binary vectors,
and aligns entity argument values to WordPiece subword BIO tags.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from .tokenizer import TOKENIZER, TAG_TO_IDX, TOOL_TO_IDX, align_tags_to_tokens


class ARNDataset(Dataset):
    """
    PyTorch Dataset for Aether Routing Network training and evaluation.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the JSONL dataset.
    tokenizer : BertTokenizerFast
        Pre-trained WordPiece fast tokenizer.
    max_seq_len : int
        Maximum sequence length for tokenization padding/truncation.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Optional[BertTokenizerFast] = None,
        max_seq_len: int = 64,
        max_length: Optional[int] = None
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {jsonl_path}")

        self.tokenizer = tokenizer if tokenizer is not None else TOKENIZER
        self.max_seq_len = max_length if max_length is not None else max_seq_len
        self.samples: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parsed = self._parse_entry(obj)
                    if parsed is not None:
                        self.samples.append(parsed)
                except Exception:
                    continue

    def _parse_entry(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse raw JSONL entry into user text and expected tools/slots."""
        messages = obj.get("messages", [])
        user_text = ""
        assistant_content = ""

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                user_text = msg.get("content", "").strip()
            elif role == "assistant":
                assistant_content = msg.get("content", "").strip()

        if not user_text:
            return None

        # Parse tool calls from assistant message
        tool_calls: List[Dict[str, Any]] = []
        if assistant_content:
            try:
                parsed_content = json.loads(assistant_content)
                if isinstance(parsed_content, list):
                    tool_calls = parsed_content
                elif isinstance(parsed_content, dict):
                    tool_calls = [parsed_content]
            except Exception:
                pass

        # Identify active tools
        active_tools: List[str] = []
        arguments: Dict[str, str] = {}
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name", "")
            if name and name in TOOL_TO_IDX:
                active_tools.append(name)
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                for k, v in args.items():
                    if v is not None and str(v).strip():
                        arguments[k] = str(v).strip()

        # Word tokenization for word-level BIO tag alignment
        words = user_text.split()
        if not words:
            return None

        word_tags = ["O"] * len(words)

        # Match argument values to word spans
        for slot_name, slot_val in arguments.items():
            val_words = slot_val.split()
            if not val_words:
                continue

            val_len = len(val_words)
            val_lower = [w.lower().strip("',.\"?;:") for w in val_words]

            for i in range(len(words) - val_len + 1):
                window_words = [words[i + k].lower().strip("',.\"?;:") for k in range(val_len)]
                if window_words == val_lower:
                    b_tag = f"B-{slot_name}"
                    i_tag = f"I-{slot_name}"
                    if b_tag in TAG_TO_IDX:
                        word_tags[i] = b_tag
                        for k in range(1, val_len):
                            word_tags[i + k] = i_tag if i_tag in TAG_TO_IDX else "O"
                    break

        return {
            "text": user_text,
            "words": words,
            "word_tags": word_tags,
            "active_tools": active_tools,
            "group_id": obj.get("group_id", None)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Align word tags to subtoken IDs
        input_ids, attention_mask, aligned_tag_ids = align_tags_to_tokens(
            sample["words"],
            sample["word_tags"],
            self.tokenizer,
            max_seq_len=self.max_seq_len
        )

        # Binary multi-label tool vector
        num_tools = len(TOOL_TO_IDX)
        tool_labels = torch.zeros(num_tools, dtype=torch.float32)
        for tool_name in sample["active_tools"]:
            if tool_name in TOOL_TO_IDX:
                tool_labels[TOOL_TO_IDX[tool_name]] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "tags": torch.tensor(aligned_tag_ids, dtype=torch.long),
            "tag_ids": torch.tensor(aligned_tag_ids, dtype=torch.long),
            "tool_labels": tool_labels
        }
