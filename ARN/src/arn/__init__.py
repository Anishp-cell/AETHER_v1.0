"""
Aether Routing Network (ARN)
A Sub-18MB Non-Generative Joint Encoder-CRF Model for Real-Time On-Device Tool Orchestration.
"""

from .config import ARNConfig
from .model import AetherRoutingNetwork
from .far import FactoredAttentionRouter
from .crf import CRF
from .loss import ARNLoss
from .dataset import ARNDataset
from .inference import predict, extract_slot_arguments
from .baselines import JointBERT, CLSCRFModel, FARNoCRFModel
from .tokenizer import TOKENIZER, TOOL_TO_IDX, IDX_TO_TOOL, TAG_TO_IDX, IDX_TO_TAG

__version__ = "0.2.0"

__all__ = [
    "ARNConfig",
    "AetherRoutingNetwork",
    "JointBERT",
    "CLSCRFModel",
    "FARNoCRFModel",
    "FactoredAttentionRouter",
    "CRF",
    "ARNLoss",
    "ARNDataset",
    "predict",
    "extract_slot_arguments",
    "TOKENIZER",
    "TOOL_TO_IDX",
    "IDX_TO_TOOL",
    "TAG_TO_IDX",
    "IDX_TO_TAG",
]
