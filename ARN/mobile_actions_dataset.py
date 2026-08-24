import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizerFast

from mobile_actions_config import (
    ARNMobileConfig, MOBILE_TOOL_TO_IDX, MOBILE_TAG_TO_IDX,
    IDX_TO_MOBILE_TOOL, IDX_TO_MOBILE_TAG, MOBILE_SLOT_TYPES
)

class MobileActionsDataset(Dataset):
    """
    Dataset adapter for google/mobile-actions with in-memory pre-encoding.
    Pre-encodes all rows at initialization time for maximum throughput.
    """
    def __init__(self, split_type: str = "train", tokenizer = None, max_seq_len: int = 64):
        self.tokenizer = tokenizer or BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_seq_len = max_seq_len
        
        # Load and filter raw dataset
        raw_ds = load_dataset("google/mobile-actions", split="train")
        self.ds = raw_ds.filter(lambda x: x["metadata"] == split_type)
        print(f"  [MobileActionsDataset] Loaded {len(self.ds)} rows for split='{split_type}'. Pre-encoding in memory...", flush=True)

        self.temporal_pattern = re.compile(
            r'(?:(?:on|for|at|this|next|upcoming|coming)\s+)?'
            r'(?:(?:this|next|upcoming|coming)\s+)?'
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?'
            r'[\s,]*'
            r'(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
            r'[\s,]*'
            r'(?:at\s+)?'
            r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}\s*(?:am|pm)|noon|midnight)?',
            re.IGNORECASE
        )

        # Pre-encoded tensor buffers
        self.input_ids_list = []
        self.attention_mask_list = []
        self.tool_labels_list = []
        self.tags_list = []

        for i in range(len(self.ds)):
            item = self._process_row(self.ds[i])
            self.input_ids_list.append(item["input_ids"])
            self.attention_mask_list.append(item["attention_mask"])
            self.tool_labels_list.append(item["tool_labels"])
            self.tags_list.append(item["tags"])

        # Stack into single contiguous tensors for instant zero-copy slicing
        self.input_ids = torch.stack(self.input_ids_list)
        self.attention_mask = torch.stack(self.attention_mask_list)
        self.tool_labels = torch.stack(self.tool_labels_list)
        self.tags = torch.stack(self.tags_list)
        print(f"  [MobileActionsDataset] Pre-encoding complete: {len(self.input_ids)} samples ready.", flush=True)

    def __len__(self):
        return len(self.input_ids)

    def _find_span(self, text, value):
        if not value or not isinstance(value, str):
            return None
        idx = text.find(value)
        if idx != -1:
            return idx, idx + len(value)
        lower_text = text.lower()
        lower_val = value.lower()
        idx = lower_text.find(lower_val)
        if idx != -1:
            return idx, idx + len(value)
        stripped = value.strip('\'".')
        idx = lower_text.find(stripped.lower())
        if idx != -1:
            return idx, idx + len(stripped)
        return None

    def _extract_datetime_span(self, text):
        matches = list(self.temporal_pattern.finditer(text))
        if matches:
            best = max(matches, key=lambda m: len(m.group(0)))
            return best.start(), best.end()
        return None

    def _process_row(self, row):
        msgs = row["messages"]
        user_text = msgs[1]["content"]
        tool_calls = msgs[-1].get("tool_calls") or []

        # 1. Multi-label tool vector
        tool_vector = torch.zeros(len(MOBILE_TOOL_TO_IDX), dtype=torch.float32)
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            if tool_name in MOBILE_TOOL_TO_IDX:
                tool_vector[MOBILE_TOOL_TO_IDX[tool_name]] = 1.0

        # 2. Tokenize user text into words and identify char offsets
        words = user_text.split()
        char_spans = []
        cur_pos = 0
        for w in words:
            start = user_text.find(w, cur_pos)
            end = start + len(w)
            char_spans.append((start, end))
            cur_pos = end

        # 3. Find character spans for all arguments
        word_tags = ["O"] * len(words)
        arg_spans = []
        for tc in tool_calls:
            args = tc["function"]["arguments"]
            for slot_name, slot_val in args.items():
                if slot_val is None:
                    continue
                if slot_name == "datetime":
                    dt_span = self._extract_datetime_span(user_text)
                    if dt_span:
                        arg_spans.append((dt_span[0], dt_span[1], "datetime"))
                elif slot_name in MOBILE_SLOT_TYPES:
                    span = self._find_span(user_text, str(slot_val))
                    if span:
                        arg_spans.append((span[0], span[1], slot_name))

        # 4. Map char spans to word BIO tags
        for start_c, end_c, slot_type in arg_spans:
            first_word = True
            for w_idx, (w_start, w_end) in enumerate(char_spans):
                if not (w_end <= start_c or w_start >= end_c):
                    if first_word:
                        word_tags[w_idx] = f"B-{slot_type}"
                        first_word = False
                    else:
                        word_tags[w_idx] = f"I-{slot_type}"

        # 5. Tokenize into subwords and align BIO tags
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length"
        )
        word_ids = tokenized.word_ids()
        aligned_tags = []
        prev_w_idx = None

        for w_idx in word_ids:
            if w_idx is None:
                aligned_tags.append(MOBILE_TAG_TO_IDX["<PAD>"])
                prev_w_idx = None
                continue
            w_tag = word_tags[w_idx]
            if w_idx == prev_w_idx:
                if w_tag.startswith("B-"):
                    slot = w_tag.split("-")[1]
                    aligned_tags.append(MOBILE_TAG_TO_IDX[f"I-{slot}"])
                else:
                    aligned_tags.append(MOBILE_TAG_TO_IDX[w_tag])
            else:
                aligned_tags.append(MOBILE_TAG_TO_IDX[w_tag])
            prev_w_idx = w_idx

        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "tool_labels": tool_vector,
            "tags": torch.tensor(aligned_tags, dtype=torch.long)
        }

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "tool_labels": self.tool_labels[idx],
            "tags": self.tags[idx]
        }
