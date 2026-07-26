"""
ARN Fast-Path Router for AETHER V2.0
Executes local sub-10ms tool selection and argument extraction using the 4.4M non-generative ARN model.
"""
import os
import sys
import time
import re
import json
from typing import Dict, Any, List

import torch

# Ensure ARN directory is in Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARN_DIR = os.path.join(PROJECT_ROOT, "ARN")
if ARN_DIR not in sys.path:
    sys.path.insert(0, ARN_DIR)

from config import ARNconfig
from tokenizer_utils import TOKENIZER, IDX_TO_TAG, IDX_TO_TOOL
from model import AetherRoutingNetwork


class ARNRouter:
    """
    Sub-10ms Local Fast-Path Router for AETHER V2.0.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ARNRouter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: str = None, confidence_threshold: float = 0.85):
        if self._initialized:
            return
        
        self.config = ARNconfig()
        self.tokenizer = TOKENIZER
        self.confidence_threshold = confidence_threshold
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(ARN_DIR, "checkpoints", "arn_best_model.pt")
            
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = torch.device("cpu")
        self.is_loaded = False
        
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Loads the trained 4.4M ARN model into memory."""
        try:
            if not os.path.exists(self.checkpoint_path):
                print(f"[ARN Router] Warning: Checkpoint file not found at '{self.checkpoint_path}'.")
                self.is_loaded = False
                return

            self.model = AetherRoutingNetwork(self.config)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            print(f"[ARN Router] Successfully loaded sub-18MB ARN engine from '{self.checkpoint_path}'.")
        except Exception as e:
            print(f"[ARN Router] Error loading ARN model: {e}")
            self.is_loaded = False

    def route_query(self, text: str) -> Dict[str, Any]:
        """
        Processes an incoming user query text in <10ms CPU latency.
        Returns routing decision, extracted arguments, and execution payload.
        """
        start_time = time.perf_counter()
        
        if not self.is_loaded or self.model is None:
            return {
                "is_local_route": False,
                "confidence": 0.0,
                "latency_ms": 0.0,
                "tools": [],
                "arguments": {},
                "tool_calls": [],
                "reason": "ARN model not loaded"
            }

        inputs = self.tokenizer(
            text,
            max_length=self.config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state
            
            far_logits, routing_weights = self.model.far(features, attention_mask)
            tool_probs = torch.sigmoid(far_logits)[0]
            
            far_threshold = getattr(self.config, "FAR_THRESHOLD", 0.5)
            tool_preds = (tool_probs >= far_threshold).int()

            active_indices = torch.where(tool_preds)[0].tolist()
            predicted_tools = [IDX_TO_TOOL[idx] for idx in active_indices]

            emissions = self.model.slot_tagger(features)
            tag_preds = self.model.crf.viterbi_decode(emissions, attention_mask)
            predicted_tag_ids = tag_preds[0]


        # Calculate max and mean FAR confidence for active predictions
        if len(active_indices) > 0:
            max_prob = float(torch.max(tool_probs[active_indices]).item())
            mean_prob = float(torch.mean(tool_probs[active_indices]).item())
            confidence = max_prob
        else:
            max_prob = 0.0
            mean_prob = 0.0
            confidence = 0.0

        # Decode token BIO argument spans
        subtokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        arguments = {}
        current_tag_base = None
        current_tokens = []

        def _save_argument():
            if current_tag_base is not None and current_tokens:
                val_str = " ".join(current_tokens)
                clean_val = re.sub(r'\s+##', '', val_str).strip()
                arguments[current_tag_base] = clean_val

        for i in range(len(subtokens)):
            token = subtokens[i]
            tag = IDX_TO_TAG[predicted_tag_ids[i]]

            if tag == "<PAD>" or token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if tag.startswith("B-"):
                if current_tag_base is not None:
                    _save_argument()
                current_tag_base = tag.split("-", 1)[1]
                current_tokens = [token]

            elif tag.startswith("I-"):
                tag_type = tag.split("-", 1)[1]
                if current_tag_base == tag_type:
                    current_tokens.append(token)

            elif tag == "O":
                if current_tag_base is not None:
                    _save_argument()
                current_tag_base = None
                current_tokens = []

        if current_tag_base is not None:
            _save_argument()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build tool calls payload compatible with AETHER tools_executor
        tool_calls = []
        for tool_name in predicted_tools:
            tool_args = {}
            for tag_key, val in arguments.items():
                # Direct tool-specific parameter mapping
                if tool_name == "send_whatsapp_message":
                    if tag_key == "contact":
                        tool_args["contact_name"] = val
                    elif tag_key == "message":
                        tool_args["message"] = val
                elif tool_name == "open_app_and_type":
                    if tag_key == "app":
                        tool_args["app_name"] = val
                    elif tag_key in ["message", "query", "value"]:
                        tool_args["text_to_type"] = val
                elif tool_name == "set_timer":
                    if tag_key == "value" and val.isdigit():
                        tool_args["minutes"] = float(val)
                    elif tag_key in ["reminder", "message"]:
                        tool_args["reminder_message"] = val
                elif tool_name == "open_url":
                    if tag_key in ["url", "target"]:
                        tool_args["url"] = val
                elif tool_name in ["search_web", "search_and_read_web"]:
                    if tag_key in ["query", "message", "target"]:
                        tool_args["query"] = val
                elif tool_name == "handle_smart_home":
                    if tag_key == "device":
                        tool_args["device"] = val
                    elif tag_key == "action":
                        tool_args["action"] = val
                elif tool_name == "media_control":
                    if tag_key == "action":
                        tool_args["action"] = val
                    elif tag_key == "value" and val.isdigit():
                        tool_args["value"] = int(val)
                elif tool_name == "run_computer_command":
                    if tag_key == "action":
                        tool_args["action_type"] = val
                    elif tag_key == "target":
                        tool_args["target"] = val
                else:
                    # Fallback mapping
                    if tag_key.startswith(f"{tool_name}_"):
                        param_name = tag_key[len(f"{tool_name}_"):]
                        tool_args[param_name] = val
                    elif "_" not in tag_key:
                        tool_args[tag_key] = val

            # Default fallbacks if required fields missing
            if tool_name == "send_whatsapp_message":
                if "contact_name" not in tool_args:
                    contact_match = re.search(r'(?:to|contact)\s+([A-Za-z0-9\s]+?)\s+(?:saying|that|telling|msg|message|say|$)', text, re.IGNORECASE)
                    if contact_match:
                        tool_args["contact_name"] = contact_match.group(1).strip()
                if "message" not in tool_args:
                    msg_match = re.search(r'(?:saying|that|telling|message|msg)\s+(.+)$', text, re.IGNORECASE)
                    tool_args["message"] = msg_match.group(1).strip() if msg_match else text

            if tool_name == "set_timer" and "reminder_message" not in tool_args:
                tool_args["reminder_message"] = "Timer complete"

            tool_calls.append({
                "name": tool_name,
                "arguments": tool_args
            })



        # Hybrid local vs cloud decision rule
        is_local = (
            len(predicted_tools) > 0 and 
            confidence >= self.confidence_threshold
        )

        return {
            "is_local_route": is_local,
            "confidence": round(confidence, 4),
            "latency_ms": round(elapsed_ms, 2),
            "tools": predicted_tools,
            "arguments": arguments,
            "tool_calls": tool_calls
        }


# Global singleton instance for quick importing
ARN_ROUTER = ARNRouter()
