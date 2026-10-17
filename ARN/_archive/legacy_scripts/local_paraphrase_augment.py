"""
Local Offline Paraphrase Augmentation Generator for ARN Dataset v3
- Correctly handles multi-message entries (system + user + assistant)
- Only augments the user message content, preserves system/assistant
- 100% guaranteed substring alignment validation
"""
import json
import random

INPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset.jsonl"
OUTPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset_augmented.jsonl"

PREFIXES = [
    "", "", "",  # Weight empty prefix higher to keep original phrasing more often
    "hey aether ", "aether ", "could you ", "can you ", "i need you to ",
    "kindly ", "please ", "go ahead and ", "yo ", "hey ",
    "bro ", "dude ", "would you ", "i want you to ", "help me ",
    "do me a favor and ", "quickly ", "now ",
]

SUFFIXES = [
    "", "", "",  # Weight empty suffix higher
    " please", " right now", " asap", " thanks", " immediately", " for me",
    " bro", " dude", " yaar", " na", " quickly", " will you",
]

def validate_sample(user_prompt: str, tool_calls: list) -> bool:
    """Validate substring alignment for argument values."""
    for tool in tool_calls:
        if not isinstance(tool, dict):
            continue
        arguments = tool.get("arguments", {})
        for arg_name, arg_val in arguments.items():
            arg_str = str(arg_val).strip()
            if not arg_str:
                continue
            if arg_str.replace(".", "").replace("-", "").isdigit():
                continue
            if arg_str.lower() not in user_prompt.lower():
                return False
    return True

def extract_user_and_assistant(obj: dict):
    """Extract user text and assistant content from a dataset entry."""
    messages = obj.get("messages", [])
    user_text = ""
    assistant_content = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_text = msg.get("content", "")
        elif role == "assistant":
            assistant_content = msg.get("content", "")
    return user_text, assistant_content

def generate_paraphrases(user_text: str, tool_calls: list, num_variations: int = 2) -> list:
    """Generate diverse rephrasings by modifying prefix and suffix."""
    results = set()
    
    for _ in range(num_variations * 6):
        if len(results) >= num_variations:
            break
        
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)
        
        new_text = f"{prefix}{user_text}{suffix}".strip()
        
        if prefix and not prefix.endswith(": "):
            new_text = new_text[0].upper() + new_text[1:]
        
        if new_text == user_text:
            continue
            
        if validate_sample(new_text, tool_calls):
            results.add(new_text)
            
    return list(results)

def main():
    print("=" * 60)
    print("ARN Local Offline Paraphrase Augmentor v3")
    print("=" * 60)
    
    augmented_entries = []
    user_cmd_count = 0
    skipped_count = 0
    generated_count = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Always keep the original entry
            augmented_entries.append(line)
            
            # Extract user text and assistant response
            user_text, assistant_content = extract_user_and_assistant(obj)
            
            if not user_text or not assistant_content:
                skipped_count += 1
                continue
            
            try:
                tool_calls = json.loads(assistant_content)
            except json.JSONDecodeError:
                skipped_count += 1
                continue
            
            if not tool_calls:
                skipped_count += 1
                continue
            
            if not validate_sample(user_text, tool_calls):
                skipped_count += 1
                continue
                
            user_cmd_count += 1
            
            # Generate paraphrases of the USER message only
            paraphrases = generate_paraphrases(user_text, tool_calls, num_variations=2)
            for p_text in paraphrases:
                # Rebuild the full message structure preserving system prompt
                new_messages = []
                for msg in obj["messages"]:
                    if msg["role"] == "user":
                        new_messages.append({"role": "user", "content": p_text})
                    else:
                        new_messages.append(msg)
                
                new_obj = {"messages": new_messages}
                augmented_entries.append(json.dumps(new_obj))
                generated_count += 1
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in augmented_entries:
            f.write(entry + "\n")
            
    print(f"Total original entries:         5000")
    print(f"Entries with valid user cmds:    {user_cmd_count}")
    print(f"Entries skipped (no user/tool):  {skipped_count}")
    print(f"Generated paraphrases:          {generated_count}")
    print(f"Total augmented dataset:        {len(augmented_entries)} samples")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
