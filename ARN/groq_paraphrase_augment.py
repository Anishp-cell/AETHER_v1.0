"""
Groq API Multi-Key Rotator Paraphrase Generator for ARN Dataset
Uses Llama-3.3-70b / Llama-3.1-8b on Groq with multi-key rotation.
Provides ultra-fast, high-diversity LLM paraphrasing with 100% substring alignment validation.
"""
import json
import urllib.request
import urllib.error
import time
import random
import os
import re

# Paste your Groq API keys (gsk_...) into this list:
GROQ_API_KEYS = [
    # "gsk_...",
    # "gsk_...",
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"  # High quality 70B model

INPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset.jsonl"
OUTPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset_augmented.jsonl"

NUM_PARAPHRASES = 2
BATCH_SIZE = 5

class KeyRotator:
    def __init__(self, keys: list):
        self.keys = [k.strip() for k in keys if k.strip() and not k.strip().startswith("#")]
        self.current_idx = 0
    
    def get_key(self) -> str:
        if not self.keys:
            return None
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key

def call_groq(prompt: str, rotator: KeyRotator) -> str:
    key = rotator.get_key()
    if not key:
        return None
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        },
        method="POST"
    )
    
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        print(f"  Groq API Error {e.code}: {e.read().decode()[:150]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def validate_sample(user_prompt: str, tool_calls: list) -> bool:
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

def extract_tool_args_text(tool_calls: list) -> dict:
    args = {}
    for tool in tool_calls:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name", "")
        for arg_name, arg_val in tool.get("arguments", {}).items():
            args[f"{tool_name}.{arg_name}"] = str(arg_val)
    return args

def build_paraphrase_prompt(samples: list) -> str:
    prompt = f"""You are a data augmentation assistant. For each user command below, generate {NUM_PARAPHRASES} diverse paraphrases.

CRITICAL RULES:
1. The paraphrases must preserve ALL argument values EXACTLY as written.
2. Only change the surrounding phrasing, NOT the argument values.
3. Use diverse vocabulary: casual, formal, imperative, shorthand.
4. Each paraphrase MUST contain the argument values verbatim.

"""
    for i, (user_text, tool_calls) in enumerate(samples):
        args = extract_tool_args_text(tool_calls)
        args_str = ", ".join(f'"{k}": "{v}"' for k, v in args.items())
        prompt += f"""SAMPLE {i+1}:
Original: "{user_text}"
Arguments that MUST appear verbatim: {{{args_str}}}
"""
    
    prompt += f"""
Respond in JSON format like:
{{"1": ["paraphrase1", "paraphrase2"], "2": ["paraphrase1", "paraphrase2"]}}
"""
    return prompt

def main():
    print("=" * 60)
    print("Groq API Multi-Key Paraphrase Generator (Llama-3.3-70b)")
    print("=" * 60)
    
    rotator = KeyRotator(GROQ_API_KEYS)
    if not rotator.keys:
        print("\n⚠️  GROQ_API_KEYS list is empty!")
        print("Please edit groq_paraphrase_augment.py and add your Groq API keys (gsk_...) to the GROQ_API_KEYS list.")
        return

    print(f"Loaded {len(rotator.keys)} Groq API key(s) for rotation")
    print(f"Model: {MODEL_NAME}")
    
    original_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            messages = obj.get("messages", [])
            user_text = ""
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_text = msg["content"]
                elif msg.get("role") == "assistant":
                    assistant_content = msg["content"]
            
            try:
                tool_calls = json.loads(assistant_content)
            except json.JSONDecodeError:
                continue
            
            if tool_calls and user_text and validate_sample(user_text, tool_calls):
                original_samples.append((user_text, tool_calls))
    
    print(f"Loaded {len(original_samples)} valid original samples")
    
    augmented_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                augmented_samples.append(line)
    
    generated_count = 0
    validation_failed = 0
    
    total_batches = (len(original_samples) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Generating paraphrases across {total_batches} batches...")
    
    for batch_start in range(0, len(original_samples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(original_samples))
        batch = original_samples[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        
        prompt = build_paraphrase_prompt(batch)
        response_text = call_groq(prompt, rotator)
        
        if not response_text:
            continue
        
        try:
            parsed = json.loads(response_text)
            for i, (orig_text, tool_calls) in enumerate(batch):
                sample_key = str(i + 1)
                paras = parsed.get(sample_key, [])
                for p in paras:
                    if isinstance(p, str) and p.strip() and validate_sample(p, tool_calls):
                        entry = {
                            "messages": [
                                {"role": "user", "content": p},
                                {"role": "assistant", "content": json.dumps(tool_calls)}
                            ]
                        }
                        augmented_samples.append(json.dumps(entry))
                        generated_count += 1
                    else:
                        validation_failed += 1
        except Exception:
            pass
        
        # Small delay between batches to stay under RPM per key
        time.sleep(0.2)
        
        if batch_num % 50 == 0 or batch_num == total_batches:
            print(f"  [Batch {batch_num}/{total_batches}] Generated {generated_count} clean paraphrases so far...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in augmented_samples:
            f.write(line + "\n")
            
    print(f"\n{'='*60}")
    print("GROQ AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Original samples:     {len(original_samples)}")
    print(f"  Generated paraphrases: {generated_count}")
    print(f"  Total dataset size:   {len(augmented_samples)}")
    print(f"  Output saved to:      {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
