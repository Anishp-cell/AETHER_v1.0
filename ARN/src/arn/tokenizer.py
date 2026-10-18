"""
Tokenizer and Registry Utilities for ARN (Aether Routing Network).

Manages:
- Fast subword tokenization via BertTokenizerFast
- Tool registry (K=18 tools, closed-world schema)
- BIO slot tag registry (M=26 tags including padding)
- Subword-to-word span alignment
"""

import os
import sys
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Any

import os
import sys
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Any

# Guard against broken/incompatible torchvision imports in transformers
if "torchvision" not in sys.modules:
    class DummyMeta(type):
        def __getattr__(cls, name: str) -> int:
            return 0

    class DummyInterpolationMode(metaclass=DummyMeta):
        BILINEAR = 2
        BICUBIC = 3
        NEAREST = 0
        NEAREST_EXACT = 0

    class MockModule(ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            sub = MockModule(f"{self.__name__}.{name}")
            sub.__path__ = []
            sub.__spec__ = os.__spec__
            setattr(self, name, sub)
            sys.modules[f"{self.__name__}.{name}"] = sub
            return sub

    tv_mock = MockModule("torchvision")
    tv_mock.__path__ = []
    tv_mock.__spec__ = os.__spec__
    sys.modules["torchvision"] = tv_mock

    tv_transforms = MockModule("torchvision.transforms")
    tv_transforms.__path__ = []
    tv_transforms.__spec__ = os.__spec__
    tv_transforms.InterpolationMode = DummyInterpolationMode
    tv_mock.transforms = tv_transforms
    sys.modules["torchvision.transforms"] = tv_transforms

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import BertTokenizerFast

# ── 1. Standard Closed Tool Registry (K=18) ──────────────────────────────────
TOOL_REGISTRY: List[str] = [
    "get_current_time",
    "handle_smart_home",
    "route_to_deepseek",
    "open_app_and_type",
    "open_url",
    "search_web",
    "run_computer_command",
    "analyze_screen_with_llava",
    "search_and_read_web",
    "read_specific_url",
    "write_and_run_script",
    "get_system_diagnostics",
    "media_control",
    "send_whatsapp_message",
    "set_timer",
    "teach_new_skill",
    "browser_task",
    "browser_open_and_interact",
]

TOOL_TO_IDX: Dict[str, int] = {tool: idx for idx, tool in enumerate(TOOL_REGISTRY)}
IDX_TO_TOOL: Dict[int, str] = {idx: tool for tool, idx in TOOL_TO_IDX.items()}

# ── 2. BIO Slot Tag Registry (M=26) ──────────────────────────────────────────
TAG_REGISTRY: List[str] = [
    "O",
    "B-contact", "I-contact",
    "B-url", "I-url",
    "B-app", "I-app",
    "B-device", "I-device",
    "B-query", "I-query",
    "B-message", "I-message",
    "B-action", "I-action",
    "B-target", "I-target",
    "B-value", "I-value",
    "B-reminder", "I-reminder",
    "B-site", "I-site",
    "B-browser_task", "I-browser_task",
    "<PAD>",
]

TAG_TO_IDX: Dict[str, int] = {tag: idx for idx, tag in enumerate(TAG_REGISTRY)}
IDX_TO_TAG: Dict[int, str] = {idx: tag for tag, idx in TAG_TO_IDX.items()}

# ── 3. Canonical Argument to Slot Mapping ────────────────────────────────────
ARG_TO_SLOT_BASE: Dict[str, str] = {
    # Standard tool arguments -> canonical slot base
    "contact_name": "contact",
    "contact": "contact",
    "message": "message",
    "text_to_type": "message",
    "url": "url",
    "app_name": "app",
    "app": "app",
    "device": "device",
    "action": "action",
    "action_type": "action",
    "target": "target",
    "query": "query",
    "task_query": "query",
    "minutes": "value",
    "value": "value",
    "reminder_message": "reminder",
    "reminder": "reminder",
    "instruction": "message",
    "skill_name": "target",
    "description": "message",
    # Browser automation tool arguments
    "task": "browser_task",
    "site": "site",
    "interact_task": "browser_task",
    "browser_task": "browser_task",
}

# ── 4. Tokenizer Instance ───────────────────────────────────────────────────
def get_tokenizer(model_name: str = "google/bert_uncased_L-2_H-128_A-2") -> BertTokenizerFast:
    """Load and return the fast WordPiece tokenizer."""
    try:
        return BertTokenizerFast.from_pretrained(model_name)
    except Exception:
        return BertTokenizerFast.from_pretrained("bert-base-uncased")


TOKENIZER = get_tokenizer()


def align_tags_to_tokens(
    words: List[str],
    tags: List[str],
    tokenizer: BertTokenizerFast,
    max_seq_len: int = 64
) -> Tuple[List[int], List[int], List[int]]:
    """
    Align word-level BIO tags to WordPiece subtokens.

    First subtoken of an entity word retains the 'B-' or 'I-' tag.
    Subsequent continuation subtokens receive the matching 'I-' tag.
    Special tokens ([CLS], [SEP]) and [PAD] receive <PAD> tag index.
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length"
    )

    word_ids = tokenized.word_ids()
    attn_mask = tokenized["attention_mask"]
    aligned_tag_ids: List[int] = []
    prev_word_idx: Optional[int] = None

    pad_idx = TAG_TO_IDX["<PAD>"]
    o_idx = TAG_TO_IDX["O"]

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            # Boundary tokens ([CLS], [SEP]) within active sequence receive 'O'.
            # Padding positions beyond [SEP] (attn_mask == 0) receive '<PAD>'.
            if attn_mask[i] == 1:
                aligned_tag_ids.append(o_idx)
            else:
                aligned_tag_ids.append(pad_idx)
            prev_word_idx = None
            continue

        word_tag = tags[word_idx]
        if word_idx == prev_word_idx:
            # Word continuation subtoken: promote B- to I-
            if word_tag.startswith("B-"):
                slot_type = word_tag.split("-")[1]
                aligned_tag_ids.append(TAG_TO_IDX.get(f"I-{slot_type}", pad_idx))
            else:
                aligned_tag_ids.append(TAG_TO_IDX.get(word_tag, pad_idx))
        else:
            aligned_tag_ids.append(TAG_TO_IDX.get(word_tag, pad_idx))

        prev_word_idx = word_idx

    return tokenized["input_ids"], tokenized["attention_mask"], aligned_tag_ids

