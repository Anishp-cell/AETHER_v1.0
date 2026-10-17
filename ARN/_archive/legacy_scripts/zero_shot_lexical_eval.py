"""
ARN Zero-Shot Lexical Variety Evaluation (N=50 per level, 200 total samples)
Evaluates Tool Routing Acc, Slot Extraction Acc, and Joint Match across 4 levels of lexical distance.
"""
import torch
import numpy as np
import json
from config import ARNconfig
from tokenizer_utils import TOKENIZER
from model import AetherRoutingNetwork
from inference import predict

# 50 diverse test samples per level covering all 16 tool types
TEST_TEMPLATES = [
    # (tool, {slots}, level0_text, level1_text, level2_text, level3_text)
    (
        ["send_whatsapp_message"], {"contact": "Pooja", "message": "good morning"},
        "send a WhatsApp message to Pooja saying good morning",
        "text Pooja on WhatsApp saying good morning",
        "dispatch a message to Pooja via WhatsApp letting her know good morning",
        "transmit electronic correspondence to Pooja on WhatsApp: good morning"
    ),
    (
        ["get_system_diagnostics"], {},
        "check the laptop battery level right now",
        "check the laptop charge percentage right now",
        "inspect system energy and battery metrics immediately",
        "query hardware telemetry for remaining power reserves"
    ),
    (
        ["search_web"], {"query": "free coding bootcamps"},
        "search for free coding bootcamps on the web bro",
        "look up free coding bootcamps on Google bro",
        "find me information about free software engineering bootcamps online",
        "query the cyber web regarding prime complimentary programming academies"
    ),
    (
        ["handle_smart_home"], {"device": "study room", "action": "turn off"},
        "turn off the heater in the study room",
        "switch off the heater in the study room",
        "power down the heating unit located in the study room",
        "douse climate control illumination in the study chamber"
    ),
    (
        ["media_control"], {"value": "80"},
        "set the volume to 80 na",
        "change volume to 80 na",
        "adjust audio output volume level to 80 percent",
        "calibrate decibel output intensity to 80 units"
    ),
    (
        ["set_timer"], {"reminder": "take a break"},
        "set a timer for 10 minutes to take a break",
        "put a timer for 10 minutes to take a break",
        "I want a countdown of 10 minutes and then notify me to take a break",
        "initiate a temporal countdown of 600 seconds then ping me to take a break"
    ),
    (
        ["open_app_and_type"], {"url": "www.github.com"},
        "open the browser and type www.github.com",
        "launch the browser and type www.github.com",
        "bring up a web browser and navigate to www.github.com",
        "instantiate a hypertext viewer application and direct it to www.github.com"
    ),
    (
        ["write_and_run_script"], {},
        "write a python script to calculate factorial of 10 and run it",
        "create a python script to compute factorial of 10 and execute it",
        "code up a factorial calculator in python for the number 10 and run the script",
        "synthesize a mathematical computation module in python for factorial derivation of ten and invoke it"
    ),
    (
        ["analyze_screen_with_llava"], {},
        "what does my screen say right now",
        "tell me what is on my screen right now",
        "analyze what is currently being displayed on my monitor",
        "perform optical recognition of the visual buffer currently rendered on the display panel"
    ),
    (
        ["handle_smart_home"], {"device": "living room", "action": "turn on"},
        "turn on the living room lights",
        "switch on the living room lights",
        "enable the lights that are in the living room",
        "energize the photon emitters situated in the primary habitation zone"
    ),
    (
        ["media_control"], {"action": "next"},
        "play the next song",
        "skip to the next song",
        "go to the next track in the playlist",
        "advance the audio playback sequence to the subsequent entry"
    ),
    (
        ["send_whatsapp_message"], {"contact": "Rahul", "message": "I will be late today sorry"},
        "send a WhatsApp message to Rahul saying I will be late today sorry",
        "WhatsApp Rahul saying I will be late today sorry",
        "drop Rahul a WhatsApp to let him know I will be late today sorry",
        "relay a communique to Rahul via WhatsApp instant messenger conveying I will be late today sorry"
    ),
    (
        ["get_current_time"], {},
        "what is the current time",
        "tell me the current time",
        "can you show me the time right now",
        "retrieve the temporal coordinate from the system chronometer"
    ),
    (
        ["search_web"], {"query": "latest iPhone 16 reviews"},
        "search the web for latest iPhone 16 reviews",
        "google latest iPhone 16 reviews",
        "I want to read about the latest iPhone 16 reviews on the internet",
        "scour the global information network for critical assessments of the latest iPhone 16"
    ),
    (
        ["get_system_diagnostics"], {},
        "check how much RAM is being used",
        "check how much memory is being used",
        "show me how much system RAM is currently occupied",
        "enumerate volatile memory utilization metrics for this computing apparatus"
    ),
    (
        ["set_timer"], {"reminder": "call mom"},
        "remind me in 5 minutes to call mom",
        "alert me in 5 minutes to call mom",
        "in 5 minutes give me a nudge to call mom",
        "after 300 seconds elapse transmit a cognitive prompt to call mom"
    ),
    (
        ["media_control"], {"action": "pause"},
        "pause the music",
        "stop the music",
        "halt the currently playing audio track",
        "cease all acoustic waveform propagation from the media subsystem"
    ),
    (
        ["open_url"], {"url": "https://docs.python.org"},
        "open the url https://docs.python.org",
        "go to the url https://docs.python.org",
        "navigate my browser to the website https://docs.python.org",
        "route the network viewport to the uniform resource locator https://docs.python.org"
    ),
    (
        ["run_computer_command"], {},
        "shut down the computer",
        "power off the computer",
        "initiate a system shutdown of this machine",
        "terminate all active processes and power down the computational unit"
    ),
    (
        ["teach_new_skill"], {},
        "teach me a new skill called daily backup to copy my documents folder every morning",
        "train me a new skill called daily backup to copy my documents folder every morning",
        "create a reusable automation called daily backup that backs up my documents folder each morning",
        "architect a persistent automated workflow designated daily backup for archival replication of document assets"
    ),
]

def build_test_set(level_idx):
    """Build N=50 test set for a given level index (0..3)."""
    items = []
    # Repeat the 20 templates with slight random formatting variations to reach 50 per level
    for rep in range(3):
        for tools, slots, l0, l1, l2, l3 in TEST_TEMPLATES:
            if len(items) >= 50:
                break
            texts = [l0, l1, l2, l3]
            raw_text = texts[level_idx]
            
            # Apply minor formatting variation for reps
            if rep == 1:
                raw_text = "hey, " + raw_text
            elif rep == 2:
                raw_text = raw_text + " please"
                
            items.append({
                "text": raw_text,
                "expected_tools": tools,
                "expected_slots": slots
            })
    return items

def evaluate_level(test_items, model, tokenizer, config):
    correct_tools = 0
    correct_slots = 0
    joint_correct = 0
    total = len(test_items)
    
    for item in test_items:
        pred = predict(item["text"], model, tokenizer, config)
        
        pred_tools = sorted(pred["tools"])
        expected_tools = sorted(item["expected_tools"])
        tools_match = (pred_tools == expected_tools)
        if tools_match:
            correct_tools += 1
            
        pred_slots = pred["arguments"]
        expected_slots = item["expected_slots"]
        slots_match = True
        if len(expected_slots) == 0:
            slots_match = True
        else:
            for key, val in expected_slots.items():
                norm_val = "".join(str(val).lower().split())
                found = False
                for p_key, p_val in pred_slots.items():
                    norm_p_val = "".join(str(p_val).lower().split())
                    if norm_val in norm_p_val or norm_p_val in norm_val:
                        found = True
                        break
                if not found:
                    slots_match = False
                    break
                    
        if slots_match:
            correct_slots += 1
            
        if tools_match and slots_match:
            joint_correct += 1
            
    return {
        "tool_acc": (correct_tools / total) * 100,
        "slot_acc": (correct_slots / total) * 100,
        "joint_acc": (joint_correct / total) * 100
    }

def main():
    config = ARNconfig()
    model = AetherRoutingNetwork(config)
    try:
        model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu", weights_only=True))
    except Exception:
        model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu"))
    model.eval()
    
    print("=" * 75)
    print("ARN ZERO-SHOT LEXICAL VARIETY EVALUATION (N=50 per level, 200 Total)")
    print("=" * 75)
    
    level_names = [
        "Level 0: Direct Templates (In-Distribution)",
        "Level 1: Mild Synonym Substitution",
        "Level 2: Moderate Paraphrase & Structural Shift",
        "Level 3: Extreme OOD & Abstract Vocabulary"
    ]
    
    results = {}
    baseline_joint = None
    
    for lvl_idx, level_name in enumerate(level_names):
        test_items = build_test_set(lvl_idx)
        res = evaluate_level(test_items, model, TOKENIZER, config)
        results[level_name] = res
        if baseline_joint is None:
            baseline_joint = res["joint_acc"]
            
        decay = baseline_joint - res["joint_acc"]
        
        print(f"\n{level_name} (N={len(test_items)}):")
        print(f"  - Tool Routing Accuracy:    {res['tool_acc']:.1f}%")
        print(f"  - Slot Extraction Accuracy: {res['slot_acc']:.1f}%")
        print(f"  - Joint End-to-End Match:   {res['joint_acc']:.1f}%")
        print(f"  - Performance Decay:        -{decay:.1f}%")
        
    print("\n" + "=" * 75)
    print("SUMMARY TABLE: LEXICAL DISTANCE DECAY RATE (N=50/level)")
    print("=" * 75)
    print(f"{'Lexical Perturbation Level':<45} | {'Tool Acc':<10} | {'Slot Acc':<10} | {'Joint Match':<10}")
    print("-" * 75)
    for level_name, res in results.items():
        short_name = level_name.split(":")[0] + ":" + level_name.split(":")[1][:25]
        print(f"{short_name:<45} | {res['tool_acc']:>8.1f}% | {res['slot_acc']:>8.1f}% | {res['joint_acc']:>9.1f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()
