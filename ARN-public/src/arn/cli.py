"""
Command Line Interface (arn-cli) for Aether Routing Network.

Usage:
    arn-cli "send an email to Alex saying meeting is delayed"
    arn-cli "turn off the living room lights" --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import torch

from .config import ARNConfig
from .model import AetherRoutingNetwork
from .tokenizer import TOKENIZER
from .inference import predict


def find_default_checkpoint() -> Optional[Path]:
    """Search for available checkpoints in order of priority."""
    search_paths = [
        Path("checkpoints/arn_final_model/model.pt"),
        Path("ARN/checkpoints/arn_final_model/model.pt"),
        Path(__file__).resolve().parent.parent.parent / "checkpoints" / "arn_final_model" / "model.pt",
        Path("checkpoints/arn_best_model.pt"),
        Path("ARN/checkpoints/arn_best_model.pt"),
        Path("_archive/checkpoints/arn_best_model.pt"),
        Path("ARN/_archive/checkpoints/arn_best_model.pt"),
        Path(__file__).resolve().parent.parent.parent / "checkpoints" / "arn_best_model.pt",
        Path(__file__).resolve().parent.parent.parent / "_archive" / "checkpoints" / "arn_best_model.pt",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="arn-cli",
        description="Aether Routing Network (ARN) On-Device Command Router & Slot Extractor."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Natural language command to route (e.g. 'turn on the bedroom lights')"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all tools and argument slot tags in the registered schema"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained ARN PyTorch checkpoint (.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON payload"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on ('cpu' or 'cuda')"
    )

    args = parser.parse_args()

    if args.list_tools:
        from .tokenizer import TOOL_TO_IDX, TAG_TO_IDX
        print("Registered Tool Registry (K=18):")
        for t, idx in sorted(TOOL_TO_IDX.items(), key=lambda x: x[1]):
            print(f"  [{idx:2d}] {t}")
        print("\nRegistered Slot Tag Inventory (|Y|=26):")
        for tag, idx in sorted(TAG_TO_IDX.items(), key=lambda x: x[1]):
            print(f"  [{idx:2d}] {tag}")
        return 0

    if not args.query:
        parser.print_help()
        return 1

    # Load configuration
    if args.config:
        config = ARNConfig.from_yaml(args.config)
    else:
        config = ARNConfig()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()
    model = None

    if ckpt_path and ckpt_path.exists():
        try:
            state = torch.load(str(ckpt_path), map_location=device)
            # Detect architecture: CLSCRFModel vs AetherRoutingNetwork
            is_cls_crf = any("tool_classifier" in k for k in state.keys())
            if is_cls_crf:
                from .baselines import CLSCRFModel
                model = CLSCRFModel(
                    num_tools=config.num_tools,
                    num_tags=config.num_tags,
                    hidden_dim=config.hidden_dim,
                    freeze_backbone=False,
                ).to(device)
                model.load_state_dict(state)
            else:
                model = AetherRoutingNetwork(config=config, load_pretrained=True).to(device)
                model.load_state_dict(state, strict=False)
        except Exception as e:
            if not args.json:
                print(f"[Warning] Could not load checkpoint from {ckpt_path}: {e}", file=sys.stderr)

    if model is None:
        model = AetherRoutingNetwork(config=config, load_pretrained=True).to(device)
        if not args.json:
            print("[Note] No trained checkpoint found. Running forward pass with initial weights.", file=sys.stderr)

    model.eval()

    # Run inference
    result = predict(args.query, model, TOKENIZER, config)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print("ARN On-Device Tool Orchestrator")
        print("=" * 65)
        print(f"Utterance:        \"{args.query}\"")
        if result["tools"]:
            print(f"Activated Tools:  {', '.join(result['tools'])}")
            print(f"Routing Prob:     {result['confidence'] * 100:.1f}%")
        else:
            print("Activated Tools:  [None - Non-Actionable / OOD]")

        if result["arguments"]:
            print("Extracted Slots:")
            for k, v in result["arguments"].items():
                print(f"  - {k:<15s}: \"{v}\"")
        else:
            print("Extracted Slots:  None")

        if result["cloud_fallback"]:
            print("Routing Decision: [FALLBACK -> ROUTE TO CLOUD LLM]")
        else:
            print("Routing Decision: [LOCAL EXECUTION]")
        print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
