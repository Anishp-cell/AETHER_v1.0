import json
import torch
import numpy as np
from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, TOOL_TO_IDX
from model import AetherRoutingNetwork
from inference import predict

# ── 1. Define Out-of-Distribution (OOD) / Adversarial Test Set ─────────
# These sentences test synonyms, structural variations, and out-of-vocab phrasing.
OOD_TEST_SET = [
    {
        "text": "Can you check my mobile battery status and WhatsApp Rahul saying I will call him soon?",
        "expected_tools": ["get_system_diagnostics", "send_whatsapp_message"],
        "expected_slots": {"contact": "Rahul", "message": "I will call him soon"}
    },
    {
        "text": "Look up who won the 2026 World Cup on Google.",
        "expected_tools": ["search_web"],
        "expected_slots": {"query": "who won the 2026 World Cup"}
    },
    {
        "text": "Terminate all log folders using rm -rf var log structural entries.",
        "expected_tools": ["run_computer_command"],
        "expected_slots": {"target": "rm -rf var log structural entries"}
    },
    {
        "text": "Learn a new task named sync repository: check git logs and pull code.",
        "expected_tools": ["teach_new_skill"],
        "expected_slots": {"target": "sync repository", "message": "check git logs and pull code"}
    },
    {
        "text": "Drop a message to Sarah... actually make it Pooja on WhatsApp telling her I am late.",
        "expected_tools": ["send_whatsapp_message"],
        "expected_slots": {"contact": "Pooja", "message": "I am late"}
    },
    {
        "text": "Search the web for Apple stock price and run k8s cluster health check.",
        "expected_tools": ["search_web", "run_computer_command"],
        "expected_slots": {"query": "Apple stock price", "target": "k8s cluster health check"}
    },
    {
        "text": "Turn off the heater in the study room, and play the music.",
        "expected_tools": ["handle_smart_home", "media_control"],
        "expected_slots": {"device": "study room", "action": "turn off"}
    },
    {
        "text": "Fetch text content from wikipedia.org and summarize it.",
        "expected_tools": ["read_specific_url"],
        "expected_slots": {"url": "wikipedia.org"}
    },
    {
        "text": "Set system volume to 60 percent.",
        "expected_tools": ["media_control"],
        "expected_slots": {"value": "60"}
    },
    {
        "text": "Remind me in 10 minutes to take a walk.",
        "expected_tools": ["set_timer"],
        "expected_slots": {"value": "10", "reminder": "take a walk"}
    }
]

# ── 2. Define In-Distribution (ID) / Easy Test Set ──────────────────────
# These are directly extracted from your clean template generator dataset distribution.
ID_TEST_SET = [
    {
        "text": "umm send a WhatsApp message to my friend Pooja saying good morning have a great day thx",
        "expected_tools": ["send_whatsapp_message"],
        "expected_slots": {"contact": "my friend Pooja", "message": "good morning have a great day"}
    },
    {
        "text": "check the laptop battery level right now",
        "expected_tools": ["get_system_diagnostics"],
        "expected_slots": {}
    },
    {
        "text": "search for best free coding bootcamps on the web bro",
        "expected_tools": ["search_web"],
        "expected_slots": {"query": "best free coding bootcamps"}
    },
    {
        "text": "teach yourself how to check my unread emails by writing use IMAP to connect to Gmail and count unread emails then show a notification for me",
        "expected_tools": ["teach_new_skill"],
        "expected_slots": {"target": "check my unread emails", "message": "use IMAP to connect to Gmail and count unread emails then show a notification"}
    },
    {
        "text": "ok run ping google.com for me",
        "expected_tools": ["run_computer_command"],
        "expected_slots": {"target": "ping google.com"}
    },
    {
        "text": "go ahead and read the content of stackoverflow.com",
        "expected_tools": ["read_specific_url"],
        "expected_slots": {"url": "stackoverflow.com"}
    },
    {
        "text": "set the volume to 80 na",
        "expected_tools": ["media_control"],
        "expected_slots": {"value": "80"}
    },
    {
        "text": "yo check the time and then WhatsApp my brother Ankit saying happy birthday bro",
        "expected_tools": ["get_current_time", "send_whatsapp_message"],
        "expected_slots": {"contact": "my brother Ankit", "message": "happy birthday bro"}
    },
    {
        "text": "quickly describe what you see on my display",
        "expected_tools": ["analyze_screen_with_llava"],
        "expected_slots": {"query": "describe what you see on my display"}
    },
    {
        "text": "please write a script to rename all files in a folder with timestamps asap",
        "expected_tools": ["write_and_run_script"],
        "expected_slots": {"message": "a script to rename all files in a folder with timestamps"}
    }
]

def evaluate_set(test_set, model, tokenizer, config, name=""):
    correct_tools = 0
    correct_slots = 0
    joint_correct = 0
    total = len(test_set)
    
    print(f"\n--- Detailed predictions for {name} ---")
    for idx, item in enumerate(test_set):
        pred = predict(item["text"], model, tokenizer, config)
        
        # 1. Evaluate Tools
        pred_tools = sorted(pred["tools"])
        expected_tools = sorted(item["expected_tools"])
        tools_match = (pred_tools == expected_tools)
        if tools_match:
            correct_tools += 1
            
        # 2. Evaluate Slots
        pred_slots = pred["arguments"]
        expected_slots = item["expected_slots"]
        
        # Clean expected slot structures for comparison
        slots_match = True
        if len(expected_slots) == 0:
            slots_match = (len(pred_slots) == 0)
        else:
            for key, val in expected_slots.items():
                # Normalize: remove all whitespace for robust subword comparison
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
            
        # Verbose output for debugging
        print(f"Sample {idx+1}: {repr(item['text'][:60])}...")
        print(f"  Expected Tools: {expected_tools} | Predicted: {pred_tools} {'[OK]' if tools_match else '[FAIL]'}")
        print(f"  Expected Slots: {expected_slots} | Predicted: {pred_slots} {'[OK]' if slots_match else '[FAIL]'}")
        print("-" * 50)
            
    return {
        "tool_accuracy": (correct_tools / total) * 100,
        "slot_accuracy": (correct_slots / total) * 100,
        "joint_accuracy": (joint_correct / total) * 100
    }

def main():
    config = ARNconfig()
    device = torch.device("cpu")
    
    # Load model
    model = AetherRoutingNetwork(config)
    try:
        model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu"))
    except FileNotFoundError:
        print("Error: checkpoints/arn_best_model.pt not found. Run train.py first.")
        return
        
    model.eval()
    
    print("="*60)
    print("ARN ADVOCATE: EVALUATING ID VS OOD GENERALIZATION")
    print("="*60)
    
    id_results = evaluate_set(ID_TEST_SET, model, TOKENIZER, config, name="Set A (In-Distribution)")
    print(f"\n[+] Set A: In-Distribution (ID) Test Split Results Summary:")
    print(f"    - Tool Selection Accuracy:  {id_results['tool_accuracy']:.1f}%")
    print(f"    - Slot Extraction Accuracy: {id_results['slot_accuracy']:.1f}%")
    print(f"    - Joint (End-to-End) Match: {id_results['joint_accuracy']:.1f}%")
    
    ood_results = evaluate_set(OOD_TEST_SET, model, TOKENIZER, config, name="Set B (Out-of-Distribution)")
    print(f"\n[+] Set B: Out-of-Distribution (OOD) / Adversarial Results Summary:")
    print(f"    - Tool Selection Accuracy:  {ood_results['tool_accuracy']:.1f}%")
    print(f"    - Slot Extraction Accuracy: {ood_results['slot_accuracy']:.1f}%")
    print(f"    - Joint (End-to-End) Match: {ood_results['joint_accuracy']:.1f}%")
    
    gap = id_results['joint_accuracy'] - ood_results['joint_accuracy']
    print("\n" + "-"*60)
    print(f"Generalization Gap (Joint Match Loss): {gap:.1f}%")
    print("-"*60)
    print("Insight for Paper: This gap shows the exact semantic boundaries of")
    print("training custom token embeddings from scratch. Moving to a tiny frozen")
    print("pretrained trunk (like MiniLM) will close this gap.")
    print("="*60)

if __name__ == "__main__":
    main()
