"""
Configuration Dataclasses for ARN.

Supports:
- Model trunk and dimension specifications
- 18-tool closed registry configuration
- Training hyperparameters
- Loading and saving from/to YAML
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .tokenizer import TOOL_REGISTRY, TAG_REGISTRY


@dataclass
class ARNConfig:
    """Core configuration for Aether Routing Network."""
    # Trunk Architecture
    model_name_or_path: str = "google/bert_uncased_L-2_H-128_A-2"
    vocab_size: int = 30522
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 64
    dropout: float = 0.1
    freeze_trunk: bool = True

    # Registries
    num_tools: int = 18
    num_tags: int = 26
    tools: List[str] = field(default_factory=lambda: list(TOOL_REGISTRY))
    tags: List[str] = field(default_factory=lambda: list(TAG_REGISTRY))

    # CRF & Tagging Head
    hard_bio_mask: bool = True  # Enforce valid BIO transition constraints in Viterbi
    use_crf: bool = True

    # Training Hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 30
    lambda_slot: float = 1.0
    seed: int = 42

    # Inference & Fallback
    activation_threshold: float = 0.5
    cloud_fallback_threshold: float = 0.85

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ARNConfig":
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return dict(self.__dict__)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)
