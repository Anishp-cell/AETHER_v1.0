from dataclasses import dataclass

@dataclass(frozen=True)
class ARNMobileConfig:
    VOCAB_SIZE: int = 30522           # BERT wordpiece vocab
    EMBED_DIM_SMALL: int = 64         # Factorized embedding input dim
    HIDDEN_DIM: int = 128             # BERT-tiny hidden dimension
    FFN_DIM: int = 512                # BERT-tiny FFN dimension
    NUM_HEADS: int = 2                # Attention heads
    NUM_LAYERS: int = 2               # Transformer encoder layers
    MAX_SEQ_LEN: int = 64             # Max tokens per utterance
    NUM_TOOLS: int = 7                # 7 Mobile-Actions tools
    DROPOUT: float = 0.1
    LR: float = 5e-4
    BATCH_SIZE: int = 32
    EPOCHS: int = 25
    LAMBDA_SLOT: float = 1.0

# 7 Tools in google/mobile-actions
MOBILE_TOOL_TO_IDX = {
    "create_calendar_event": 0,
    "send_email": 1,
    "create_contact": 2,
    "show_map": 3,
    "open_wifi_settings": 4,
    "turn_on_flashlight": 5,
    "turn_off_flashlight": 6
}
IDX_TO_MOBILE_TOOL = {v: k for k, v in MOBILE_TOOL_TO_IDX.items()}

# 10 Slot Types across the 7 tools
MOBILE_SLOT_TYPES = [
    "title", "datetime", "first_name", "last_name",
    "phone_number", "email", "to", "subject", "body", "query"
]

MOBILE_TAG_TO_IDX = {"O": 0}
idx_counter = 1
for slot in MOBILE_SLOT_TYPES:
    MOBILE_TAG_TO_IDX[f"B-{slot}"] = idx_counter
    MOBILE_TAG_TO_IDX[f"I-{slot}"] = idx_counter + 1
    idx_counter += 2
MOBILE_TAG_TO_IDX["<PAD>"] = idx_counter

IDX_TO_MOBILE_TAG = {v: k for k, v in MOBILE_TAG_TO_IDX.items()}
NUM_MOBILE_TAGS = len(MOBILE_TAG_TO_IDX)
