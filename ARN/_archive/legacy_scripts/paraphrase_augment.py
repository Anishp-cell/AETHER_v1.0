"""
Paraphrase Augmentation Generator for ARN Dataset
Uses Gemini API to generate diverse paraphrases of template-generated training data.
Each paraphrase is validated for substring alignment of argument values.
"""
import json
import urllib.request
import urllib.error
import time
import random
import os
import re

API_KEY = os.getenv("GEMINI_API_KEY", "")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

INPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset.jsonl"
OUTPUT_FILE = r"D:\python\AETHER_V1.0\ARN\aether_orchestrator_dataset_augmented.jsonl"

NUM_PARAPHRASES = 2  # 2 diverse paraphrases per original sample
BATCH_SIZE = 10       # 10 samples per API call to maximize efficiency
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 0.5  # 0.5s delay (using Gemini 2.5 Flash 1000 RPM quota)


def call_gemini(prompt: str) -> str:
    """Call Gemini API and return the text response."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 4096
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                wait = (attempt + 1) * 12
                print(f"  [429 Rate Limit] Backing off for {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API Error {e.code}: {e.read().decode()[:200]}")
                return None
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return None
    return None


def validate_sample(user_prompt: str, tool_calls: list) -> bool:
    """
    Validate that ALL argument values in tool_calls are substrings of user_prompt.
    This is the critical check that prevents the data poisoning bug from the original dataset.
    """
    for tool in tool_calls:
        if not isinstance(tool, dict):
            continue
        arguments = tool.get("arguments", {})
        for arg_name, arg_val in arguments.items():
            arg_str = str(arg_val).strip()
            if not arg_str:
                continue
            # Skip numeric-only values (minutes, etc.) - they don't need substring match
            if arg_str.replace(".", "").replace("-", "").isdigit():
                continue
            if arg_str.lower() not in user_prompt.lower():
                return False
    return True


def extract_tool_args_text(tool_calls: list) -> dict:
    """Extract a mapping of argument names to their values for prompt construction."""
    args = {}
    for tool in tool_calls:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name", "")
        for arg_name, arg_val in tool.get("arguments", {}).items():
            args[f"{tool_name}.{arg_name}"] = str(arg_val)
    return args


def build_paraphrase_prompt(samples: list) -> str:
    """Build a prompt that asks Gemini to paraphrase multiple training samples."""
    prompt = """You are a data augmentation assistant. For each user command below, generate {n} diverse paraphrases.

CRITICAL RULES:
1. The paraphrases must preserve ALL argument values EXACTLY as written (names, apps, URLs, etc.)
2. Only change the surrounding words/phrasing, NOT the argument values
3. Use diverse vocabulary: slang, formal, casual, imperative, polite request styles
4. Each paraphrase MUST contain the exact argument values as substrings
5. Output valid JSON array for each sample

""".format(n=NUM_PARAPHRASES)
    
    for i, (user_text, tool_calls) in enumerate(samples):
        args = extract_tool_args_text(tool_calls)
        args_str = ", ".join(f'"{k}": "{v}"' for k, v in args.items())
        prompt += f"""
SAMPLE {i+1}:
Original: "{user_text}"
Arguments that MUST appear verbatim: {{{args_str}}}

Output as JSON array of {NUM_PARAPHRASES} strings:
"""
    
    prompt += """
Respond with ONLY a JSON object mapping sample numbers to arrays of paraphrases, like:
{"1": ["paraphrase1", "paraphrase2", "paraphrase3"], "2": [...]}
"""
    return prompt


def parse_paraphrases(response: str, num_samples: int) -> dict:
    """Parse the Gemini response into a dict of sample_idx -> list of paraphrases."""
    # Try to extract JSON from the response
    # First try to find a JSON block
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        return {}
    
    try:
        result = json.loads(json_match.group())
        return result
    except json.JSONDecodeError:
        # Try cleaning the response
        cleaned = json_match.group()
        cleaned = cleaned.replace("```json", "").replace("```", "")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def main():
    print("=" * 60)
    print("ARN Paraphrase Augmentation Generator")
    print("=" * 60)
    
    # Load original dataset
    original_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            messages = obj["messages"]
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
            
            if not tool_calls or not user_text:
                continue
            
            # Only keep samples that already pass validation
            if validate_sample(user_text, tool_calls):
                original_samples.append((user_text, tool_calls))
    
    print(f"Loaded {len(original_samples)} valid original samples")
    
    # Prepare output file with all original samples first
    augmented_samples = []
    
    # Copy all originals
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                augmented_samples.append(line)
    
    print(f"Starting paraphrase generation ({NUM_PARAPHRASES} per sample)...")
    print(f"Batching {len(original_samples)} samples into groups of {BATCH_SIZE}")
    
    generated_count = 0
    failed_count = 0
    validation_failed = 0
    
    # Process in batches
    for batch_start in range(0, len(original_samples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(original_samples))
        batch = original_samples[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(original_samples) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing samples {batch_start+1}-{batch_end}...")
        
        # Build and send the prompt
        prompt = build_paraphrase_prompt(batch)
        response = call_gemini(prompt)
        
        if response is None:
            failed_count += len(batch) * NUM_PARAPHRASES
            print(f"  ❌ API call failed, skipping batch")
            continue
        
        # Parse paraphrases
        paraphrases = parse_paraphrases(response, len(batch))
        
        for i, (original_text, tool_calls) in enumerate(batch):
            sample_key = str(i + 1)
            sample_paraphrases = paraphrases.get(sample_key, [])
            
            for para_text in sample_paraphrases:
                if not isinstance(para_text, str) or not para_text.strip():
                    failed_count += 1
                    continue
                
                # Critical validation: all argument values must be substrings
                if validate_sample(para_text, tool_calls):
                    # Build the JSONL entry in the same format as the original dataset
                    entry = {
                        "messages": [
                            {"role": "user", "content": para_text},
                            {"role": "assistant", "content": json.dumps(tool_calls)}
                        ]
                    }
                    augmented_samples.append(json.dumps(entry))
                    generated_count += 1
                else:
                    validation_failed += 1
        
        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)
        
        # Progress report every 10 batches
        if batch_num % 10 == 0:
            print(f"\n  --- Progress: {generated_count} generated, {validation_failed} failed validation, {failed_count} API failures ---")
    
    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in augmented_samples:
            f.write(line + "\n")
    
    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Original samples:      {len(original_samples)}")
    print(f"  New paraphrases:       {generated_count}")
    print(f"  Validation failures:   {validation_failed} (correctly rejected)")
    print(f"  API failures:          {failed_count}")
    print(f"  Total dataset size:    {len(augmented_samples)}")
    print(f"  Output file:           {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
