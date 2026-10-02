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
        self.onnx_path = os.path.join(ARN_DIR, "checkpoints", "arn_model_int8.onnx")
        self.model = None
        self.ort_session = None
        self.use_onnx = False
        self.device = torch.device("cpu")
        self.is_loaded = False
        self.dynamic_skill_overrides: Dict[str, List[str]] = {}
        
        self._load_model()
        self._initialized = True

    def register_dynamic_skill(self, skill_name: str, keywords: List[str]):
        """Registers dynamic skill keywords into fast-path router overrides."""
        self.dynamic_skill_overrides[skill_name] = [kw.lower() for kw in keywords if kw]
        print(f"[ARN Router] Registered dynamic fast-path override for skill '{skill_name}'.")

    def _load_model(self):
        """Loads either the Int8 ONNX graph (<4.5MB) or PyTorch model (.pt)."""
        # 1. Try Int8 ONNX Runtime (Zero-RAM, <2ms CPU)
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                # Load lightweight PyTorch shell ONLY for CRF Viterbi decoding (0 HuggingFace downloads)
                self.model = AetherRoutingNetwork(self.config, load_pretrained=False)
                self.model.eval()

                
                # Disable ONNX telemetry logs
                options = ort.SessionOptions()
                options.log_severity_level = 3
                self.ort_session = ort.InferenceSession(self.onnx_path, options, providers=["CPUExecutionProvider"])
                self.use_onnx = True
                self.is_loaded = True
                print(f"[ARN Router] [ONNX Int8 Engine Online] Loaded quantized graph (4.33MB) from '{self.onnx_path}'.")

                return
            except Exception as e:
                print(f"[ARN Router Warning] Failed to initialize ONNX session: {e}. Falling back to PyTorch.")

        # 2. Fallback to PyTorch FP32
        try:
            if not os.path.exists(self.checkpoint_path):
                print(f"[ARN Router] Warning: Checkpoint file not found at '{self.checkpoint_path}'.")
                self.is_loaded = False
                return

            self.model = AetherRoutingNetwork(self.config)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.use_onnx = False
            self.is_loaded = True
            print(f"[ARN Router] Successfully loaded PyTorch ARN engine from '{self.checkpoint_path}'.")
        except Exception as e:
            print(f"[ARN Router] Error loading ARN model: {e}")
            self.is_loaded = False


    def route_query(self, text: str) -> Dict[str, Any]:
        """
        Processes an incoming user query text in <10ms CPU latency.
        Returns routing decision, extracted arguments, and execution payload.
        """
        start_time = time.perf_counter()

        # Browser Automation Fast-Route (<1ms)
        low = text.lower().strip()
        is_youtube = "youtube" in low and any(w in low for w in ["play", "watch", "song", "video", "track", "music"])
        is_cart_shop = any(w in low for w in ["to cart", "checkout", "add to cart"]) or (any(s in low for s in ["amazon", "flipkart", "ebay", "walmart"]) and any(w in low for w in ["buy", "order", "search", "add", "cart", "price"]))
        is_travel = any(w in low for w in ["book a flight", "book flight", "book tickets", "book ticket", "book hotel", "reserve flight"])
        is_browser = any(w in low for w in ["browser", "open website", "fill the form", "fill form", "fill out the form"]) or is_cart_shop or is_travel
        is_direct_url = any(low.startswith(p) for p in ["go to ", "open ", "visit "]) and any(d in low for d in [".com", ".org", ".net", ".io", ".ai", "http"])
        if is_youtube or is_browser or is_direct_url:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "is_local_route": True,
                "confidence": 0.99,
                "latency_ms": round(elapsed_ms, 2),
                "tools": ["browser_task"],
                "arguments": {"task": text, "headless": False},
                "tool_calls": [{
                    "name": "browser_task",
                    "arguments": {"task": text, "headless": False}
                }],
                "reason": "Fast-path browser automation intent matched"
            }
        
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

        if self.use_onnx and self.ort_session is not None:
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
            far_logits_np, emissions_np = self.ort_session.run(None, ort_inputs)
            far_logits = torch.from_numpy(far_logits_np)
            emissions = torch.from_numpy(emissions_np)

            tool_probs = torch.sigmoid(far_logits)[0]
            far_threshold = getattr(self.config, "FAR_THRESHOLD", 0.5)
            tool_preds = (tool_probs >= far_threshold).int()

            active_indices = torch.where(tool_preds)[0].tolist()
            predicted_tools = [IDX_TO_TOOL[idx] for idx in active_indices]

            tag_preds = self.model.crf.viterbi_decode(emissions, attention_mask)
            predicted_tag_ids = tag_preds[0]
        else:
            with torch.no_grad():
                outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                features = outputs.last_hidden_state
                
                far_logits, routing_weights = self.model.far(features, attention_mask)
                tool_probs = torch.sigmoid(far_logits)[0]
                
                far_threshold = getattr(self.config, "FAR_THRESHOLD", 0.5)
                tool_preds = (tool_probs >= far_threshold).int()

                active_indices = torch.where(tool_preds)[0].tolist()
                predicted_tools = [IDX_TO_TOOL[idx] for idx in active_indices]

                emissions = self.slot_tagger(features) if hasattr(self, 'slot_tagger') else self.model.slot_tagger(features)
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
                elif tool_name == "analyze_screen_with_llava":
                    if tag_key in ["query", "message", "target"]:
                        tool_args["task_query"] = val

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
                elif tool_name == "browser_task":
                    tool_args["task"] = arguments.get("browser_task") or arguments.get("query") or text
                    tool_args["headless"] = False
                elif tool_name == "browser_open_and_interact":
                    target_url = arguments.get("site") or arguments.get("url") or ""
                    target_goal = arguments.get("browser_task") or arguments.get("action") or ""
                    if not target_url:
                        m_u = re.search(r'([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)', text)
                        target_url = m_u.group(1) if m_u else "https://google.com"
                    if not target_goal:
                        target_goal = text
                    tool_args["url"] = target_url
                    tool_args["goal"] = target_goal
                    tool_args["headless"] = False
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

            if tool_name == "browser_task" and "task" not in tool_args:
                tool_args["task"] = text
                tool_args["headless"] = False

            if tool_name == "browser_open_and_interact":
                if "url" not in tool_args:
                    m_u = re.search(r'([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)', text)
                    tool_args["url"] = m_u.group(1) if m_u else "https://google.com"
                if "goal" not in tool_args:
                    tool_args["goal"] = text
            if tool_name in ["get_current_time", "get_system_diagnostics", "get_nim_usage", "get_most_visited_sites"]:
                tool_args = {}

            tool_calls.append({
                "name": tool_name,
                "arguments": tool_args
            })



        low_text = text.lower().strip()
        
        # Check dynamic auto-forged skill overrides
        for s_name, kws in self.dynamic_skill_overrides.items():
            if any(kw in low_text for kw in kws):
                return {
                    "is_local_route": True,
                    "confidence": 0.99,
                    "latency_ms": round((time.time() - start_time) * 1000.0, 2),
                    "tools": [s_name],
                    "arguments": {"query": text},
                    "tool_calls": [{"name": s_name, "arguments": {"query": text}}]
                }

        # ── HIGH-PRECISION DESKTOP & INTENT OVERRIDES ──
        # Check if the user is asking to OPEN Spotify as an application
        if re.search(r'\b(?:open|launch|start)\s+spotify\b', low_text) and not any(kw in low_text for kw in ["play ", "listen to ", "put on ", "stream "]):
            predicted_tools = ["open_app_and_type"]
            confidence = 0.99
            tool_calls = [{"name": "open_app_and_type", "arguments": {"app_name": "spotify", "text_to_type": ""}}]

        elif any(kw in low_text for kw in ["play ", "listen to ", "put on ", "stream ", "liked songs", "my playlist", "play music", "play track", "play song"]) and ("spotify" in low_text or "music" in low_text or "song" in low_text or "playlist" in low_text):
            predicted_tools = ["play_spotify_media"]
            confidence = 0.99
            query_clean = text
            # If compound command like "play X and explain Y", isolate the play portion
            if " and " in query_clean.lower():
                and_parts = re.split(r'\s+and\s+', query_clean, flags=re.IGNORECASE)
                for part in and_parts:
                    if any(kw in part.lower() for kw in ["play", "spotify", "music", "song", "listen"]):
                        query_clean = part
                        break

            # Strip command prefixes
            for p in ["play ", "search ", "listen to ", "put on ", "can you play ", "please play "]:
                if query_clean.lower().startswith(p):
                    query_clean = query_clean[len(p):].strip()
            # Strip trailing " on spotify", " in spotify"
            query_clean = re.sub(r'\s+(?:on|in|using)\s+spotify.*$', '', query_clean, flags=re.IGNORECASE).strip(". ")
            if not query_clean:
                query_clean = "liked songs"
            tool_calls = [{"name": "play_spotify_media", "arguments": {"query": query_clean}}]

        elif any(kw in low_text for kw in ["write a script", "python script", "matplotlib", "create a chart", "plot"]):
            predicted_tools = ["write_and_run_script"]
            confidence = 0.99
            tool_calls = [{"name": "write_and_run_script", "arguments": {"instruction": text}}]

        elif any(kw in low_text for kw in ["click the", "click on", "press the", "click submit", "click button", "click search"]):
            predicted_tools = ["vision_click"]
            confidence = 0.99
            # Extract clean target element name
            clean_elem = text.split("[SYSTEM INJECTED")[0].strip()
            clean_elem = re.sub(r'^(?:and\s+you\s+|can\s+you\s+|please\s+|could\s+you\s+|hey\s+aether\s*,?\s*|aether\s*,?\s*)', '', clean_elem, flags=re.IGNORECASE)
            clean_elem = re.sub(r'^(?:click\s+(?:on|the|on the)?|press\s+(?:the)?)\s*', '', clean_elem, flags=re.IGNORECASE)
            clean_elem = re.sub(r'(?:in|on)\s+(?:my|the)?\s*(?:current)?\s*(?:screen|display|window)\.?$', '', clean_elem, flags=re.IGNORECASE).strip()
            tool_calls = [{"name": "vision_click", "arguments": {"element_description": clean_elem}}]

        elif any(kw in low_text for kw in ["read the error", "read that error", "read error on screen", "read text on screen", "what does that say"]):
            predicted_tools = ["vision_read"]
            confidence = 0.99
            tool_calls = [{"name": "vision_read", "arguments": {"query": text}}]

        elif any(kw in low_text for kw in ["highlight the", "show me where", "find the start menu", "highlight start", "highlight search", "highlight"]):
            predicted_tools = ["vision_highlight"]
            confidence = 0.99
            # Extract clean target element name
            clean_elem = text.split("[SYSTEM INJECTED")[0].strip()
            clean_elem = re.sub(r'^(?:and\s+you\s+|can\s+you\s+|please\s+|could\s+you\s+|hey\s+aether\s*,?\s*|aether\s*,?\s*)', '', clean_elem, flags=re.IGNORECASE)
            clean_elem = re.sub(r'^(?:highlight\s+(?:the)?|show\s+me\s+where\s+(?:the)?|find\s+(?:the)?)\s*', '', clean_elem, flags=re.IGNORECASE)
            clean_elem = re.sub(r'(?:in|on)\s+(?:my|the)?\s*(?:current)?\s*(?:screen|display|window)\.?$', '', clean_elem, flags=re.IGNORECASE).strip()
            tool_calls = [{"name": "vision_highlight", "arguments": {"element_description": clean_elem}}]

        elif re.search(r'\b(?:snap|move|place|put|map|dock)\b.*\b(left|right|top|bottom|maximize|minimize|restore)\b', low_text) or any(kw in low_text for kw in ["snap window", "maximize window", "minimize window", "restore window"]):
            predicted_tools = ["snap_window"]
            confidence = 0.99
            dir_match = "left"
            for d in ["left", "right", "top", "bottom", "maximize", "minimize", "restore"]:
                if d in low_text:
                    dir_match = d
                    break
            tool_calls = [{"name": "snap_window", "arguments": {"direction": dir_match}}]

        elif any(kw in low_text for kw in ["tile windows", "tile all windows", "tile the windows", "tile window", "dial all windows", "dial windows", "tail all windows"]):
            predicted_tools = ["tile_windows"]
            confidence = 0.99
            tool_calls = [{"name": "tile_windows", "arguments": {}}]

        elif any(kw in low_text for kw in ["close this window", "close active window", "close window", "close the window"]):
            predicted_tools = ["close_foreground_window"]
            confidence = 0.99
            tool_calls = [{"name": "close_foreground_window", "arguments": {}}]

        elif any(kw in low_text for kw in ["tell me when", "alert me when", "watch for", "watch the screen for"]):
            predicted_tools = ["start_screen_watch"]
            confidence = 0.99
            cond = re.sub(r'^(?:tell me when|alert me when|watch for|watch the screen for)\s+(?:the)?\s*', '', text, flags=re.IGNORECASE).strip()
            tool_calls = [{"name": "start_screen_watch", "arguments": {"condition": cond}}]

        elif any(kw in low_text for kw in ["stop watching", "stop screen watch"]):
            predicted_tools = ["stop_screen_watch"]
            confidence = 0.99
            tool_calls = [{"name": "stop_screen_watch", "arguments": {}}]

        elif any(kw in low_text for kw in ["clipboard", "search clipboard", "copy text", "what is on my clipboard"]):
            predicted_tools = ["clipboard_action"]
            confidence = 0.99
            action_type = "search" if "search" in low_text else ("read" if "read" in low_text or "what" in low_text else "read")
            tool_calls = [{"name": "clipboard_action", "arguments": {"action": action_type}}]

        elif any(kw in low_text for kw in ["save this", "undo that", "run this", "go back"]):
            predicted_tools = ["smart_action"]
            confidence = 0.99
            tool_calls = [{"name": "smart_action", "arguments": {"command": text}}]

        elif any(kw in low_text for kw in ["click on", "click at", "mouse click", "click screen"]):
            predicted_tools = ["run_computer_command"]
            confidence = 0.99
            coords = re.findall(r'\d+', text)
            target_coords = f"{coords[0]},{coords[1]}" if len(coords) >= 2 else "500,500"
            tool_calls = [{"name": "run_computer_command", "arguments": {"action_type": "mouse_click", "target": target_coords}}]

        elif any(kw in low_text for kw in ["draw a box", "draw box", "highlight", "annotate"]):
            predicted_tools = ["annotate_screen"]
            confidence = 0.99
            coords = [int(c) for c in re.findall(r'\d+', text)]
            x = coords[0] if len(coords) >= 1 else 200
            y = coords[1] if len(coords) >= 2 else 200
            w = coords[2] if len(coords) >= 3 else 400
            h = coords[3] if len(coords) >= 4 else 250
            tool_calls = [{"name": "annotate_screen", "arguments": {"x": x, "y": y, "width": w, "height": h, "label": "AETHER Target"}}]

        elif any(kw in low_text for kw in [
            "book a flight", "book flight", "book tickets", "book ticket", "book hotel",
            "buy on amazon", "order on amazon", "check my cart", "add to cart",
            "fill out the form", "fill this form", "fill the form", "submit the form",
            "compare prices on", "track order on", "navigate browser"
        ]):
            predicted_tools = ["browser_task"]
            confidence = 0.99
            tool_calls = [{"name": "browser_task", "arguments": {"task": text, "headless": False}}]

        elif len(predicted_tools) == 0 or confidence < self.confidence_threshold:
            if any(kw in low_text for kw in ["system diagnostics", "system status", "telemetry", "cpu", "ram", "battery"]):
                predicted_tools = ["get_system_diagnostics"]
                confidence = 0.99
                tool_calls = [{"name": "get_system_diagnostics", "arguments": {}}]
            elif any(kw in low_text for kw in ["analyze my screen", "read my screen", "what is on my screen", "what's on my screen", "screen analysis"]):
                predicted_tools = ["analyze_screen_with_llava"]
                confidence = 0.99
                tool_calls = [{"name": "analyze_screen_with_llava", "arguments": {"task_query": "Describe what is on screen"}}]
            elif any(kw in low_text for kw in ["current time", "what time is it", "what's the time"]):
                predicted_tools = ["get_current_time"]
                confidence = 0.99
                tool_calls = [{"name": "get_current_time", "arguments": {}}]
            elif any(kw in low_text for kw in ["nim token", "nim tokens", "token usage", "nim usage", "nim quota", "tokens have i used"]):
                predicted_tools = ["get_nim_usage"]
                confidence = 0.99
                tool_calls = [{"name": "get_nim_usage", "arguments": {}}]
            elif any(kw in low_text for kw in ["visited sites", "most visited", "website history", "site history", "websites have i visited"]):
                predicted_tools = ["get_most_visited_sites"]
                confidence = 0.99
                tool_calls = [{"name": "get_most_visited_sites", "arguments": {}}]


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
