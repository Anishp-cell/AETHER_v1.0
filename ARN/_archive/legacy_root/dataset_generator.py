import os
import json
import time
import random
import requests

# OpenRouter API Key (loaded from environment)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEYS = [OPENROUTER_KEY] if OPENROUTER_KEY else []


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemma-4-31b-it:free"


TOOL_DEFINITIONS = [
    {
        "name": "get_current_time",
        "description": "Fetch the exact current time to answer queries about the time."
    },
    {
        "name": "handle_smart_home",
        "description": "Physically control local smart devices like lights.",
        "arguments": {
            "device": "string",
            "action": "string"
        }
    },
    {
        "name": "route_to_deepseek",
        "description": "MUST BE CALLED if the user asks any complex logic, hard math, deep reasoning, or physics problem.",
        "arguments": {
            "query": "string"
        }
    },
    {
        "name": "open_app_and_type",
        "description": "MUST be used when the user wants to open an app AND write/type text into it. For example: 'open notepad and write hello', 'open notepad and type an essay'.",
        "arguments": {
            "app_name": "string",
            "text_to_type": "string"
        }
    },
    {
        "name": "open_url",
        "description": "MUST be used when the user wants to navigate to a specific website like LinkedIn, YouTube, Instagram, Netflix, Gmail, or any URL.",
        "arguments": {
            "url": "string"
        }
    },
    {
        "name": "search_web",
        "description": "MUST be used when the user wants to search for something on the web, Google, or Chrome. For example: 'search for paneer recipe', 'Google how to learn python', 'look up the weather'.",
        "arguments": {
            "query": "string"
        }
    },
    {
        "name": "run_computer_command",
        "description": "DELEGATE to the Desktop Sub-Agent for simple PC actions: just opening an app (without typing), just pressing a key, just typing text into the currently focused window, or running a shell command.",
        "arguments": {
            "action_type": "string",
            "target": "string"
        }
    },
    {
        "name": "analyze_screen_with_llava",
        "description": "DELEGATE to the Vision Sub-Agent to read the user's screen. Use this when the user asks 'what am I looking at?' or 'read this text on screen'.",
        "arguments": {
            "task_query": "string"
        }
    },
    {
        "name": "search_and_read_web",
        "description": "DELEGATE to the AetherWeb Sub-Agent to silently scrape the internet to answer a real-world question.",
        "arguments": {
            "query": "string"
        }
    },
    {
        "name": "read_specific_url",
        "description": "DELEGATE to read the full text content of a specific URL.",
        "arguments": {
            "url": "string"
        }
    },
    {
        "name": "write_and_run_script",
        "description": "Write a Python script and execute it.",
        "arguments": {
            "instruction": "string"
        }
    },
    {
        "name": "get_system_diagnostics",
        "description": "DELEGATE to the OS Macro tool when the user asks for a system status report, battery level, CPU, or RAM metrics."
    },
    {
        "name": "media_control",
        "description": "DELEGATE to the OS Macro tool when the user asks to play/pause media, skip a song, mute the computer, or set the volume.",
        "arguments": {
            "action": "string",
            "value": "integer"
        }
    },
    {
        "name": "send_whatsapp_message",
        "description": "MANDATORY COMMAND: You MUST execute this tool IMMEDIATELY if the user asks you to text, msg, send a message to, or contact someone on WhatsApp.",
        "arguments": {
            "contact_name": "string",
            "message": "string"
        }
    },
    {
        "name": "set_timer",
        "description": "DELEGATE to the OS Macro tool when the user wants an alarm, timer, or verbal reminder at a future time.",
        "arguments": {
            "minutes": "number",
            "reminder_message": "string"
        }
    },
    {
        "name": "teach_new_skill",
        "description": "MANDATORY: Trigger this tool if the user asks you to learn a new tool, write a permanent skill, create a new capability, or teach yourself how to do something permanently.",
        "arguments": {
            "skill_name": "string",
            "description": "string",
            "instruction": "string"
        }
    }
]

SYSTEM_PROMPT = """You are a master dataset generator for a Reinforcement Learning AI pipeline. 
Your job is to generate synthetic training pairs mapping a human's spoken command to the EXACT JSON tool sequence required to achieve it.

**Rules for Generation**:
1. You must read the provided JSON Tool Definitions to understand what tools are available.
2. The user will specify a random scenario or theme.
3. You must generate 5 distinct pairs for that theme. 
4. Use realistic, diverse contact names (names that someone might store as their contact in WhatsApp, e.g. "John", "Sarah", "Sophie", "Mr. Davis") and relations (e.g., "my mom", "my friend", "my brother", "my manager") for contact names. DO NOT use literal placeholders like "<FRIEND_NAME>", "<SISTER_NAME>", or "<CONTACT_NAME>".
5. For all arguments (especially "instruction", "description", and "skill_name" in "teach_new_skill", and "target" in "run_computer_command"), the values in the expected JSON MUST be exact substrings copied from the "user_prompt". Do not rephrase, summarize, or change them in any way. This is critical for our keyword-alignment algorithm.
6. The output MUST be a strict JSON object with a single key called "pairs", which contains an array of 5 objects. Each object must have a `user_prompt` string and an `expected_json` string.
7. Use diverse language, typos, slang, and complex compound instructions. Include examples where users correct themselves mid-sentence.

**Format Example**:
{
  "pairs": [
    {
      "user_prompt": "Send a WhatsApp message to my manager saying I will be late.",
      "expected_json": "[{\\"name\\": \\"send_whatsapp_message\\", \\"arguments\\": {\\"contact_name\\": \\"my manager\\", \\"message\\": \\"I will be late.\\"}}]"
    }
  ]
}

**Tool Definitions**:
""" + json.dumps(TOOL_DEFINITIONS, indent=2)

SCENARIOS = [
    "Sending urgent WhatsApp messages to colleagues or family.",
    "Executing Linux shell commands for file management.",
    "Teaching a new skill to automate system backups.",
    "Performing a web search to find recent news, combined with sending a message.",
    "Using multiple tools sequentially, like searching the web and then messaging the result.",
    "User speaking with lots of slang and typos to run terminal commands.",
    "User correcting themselves mid-sentence while trying to teach a new Python skill."
]

def generate_samples(target_count=1500, batch_size=10):
    dataset_file = os.path.join(os.path.dirname(__file__), "aether_orchestrator_dataset.jsonl")
    
    total_generated = 0
    
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            total_generated = sum(1 for _ in f)
    else:
        open(dataset_file, "w", encoding="utf-8").close()
        
    if total_generated > 0:
        print(f"Resuming generation. Currently at {total_generated} pairs.")
    else:
        print("Starting dataset generation from scratch.")
        
    key_idx = 0
    
    with open(dataset_file, "a", encoding="utf-8") as f:
        while total_generated < target_count:
            scenario = random.choice(SCENARIOS)
            prompt = f"Generate {batch_size} varied command pairs for the scenario: {scenario}. Make them completely different from each other in phrasing. Some should be casual, some direct, some with typos or slang."
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.0,
                "response_format": {"type": "json_object"}
            }
            
            target_key = GROQ_API_KEYS[key_idx].strip()
            headers = {
                "Authorization": f"Bearer {target_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/aether-orchestrator", 
                "X-Title": "Aether Dataset Generator"
            }
            
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                
                if response.status_code == 429:
                    print(f"Rate limit hit on key {key_idx+1}. Rotating key and sleeping...")
                    key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                    time.sleep(2)
                    continue
                    
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                    time.sleep(2)
                    continue
                    
                raw_text = response.json()['choices'][0]['message']['content']
                
                data = json.loads(raw_text)
                pairs = data.get("pairs", [])
                
                for pair in pairs:
                    user_prompt = pair.get("user_prompt")
                    expected_json = pair.get("expected_json")
                    
                    if not user_prompt or not expected_json:
                        continue
                        
                    finetune_row = {
                        "messages": [
                            {"role": "system", "content": "You are a tool-calling orchestrator. Output ONLY a raw JSON array of tool calls. No text, no markdown, no explanation."},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": expected_json}
                        ]
                    }
                    
                    f.write(json.dumps(finetune_row) + "\n")
                    total_generated += 1
                    
                    if total_generated >= target_count:
                        break
                        
                print(f"Synthesized {total_generated}/{target_count} pairs...")
                
            except Exception as e:
                print(f"An error occurred: {e}")
                key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                time.sleep(2)
                
            time.sleep(1.0)
            
    print(f"Dataset generation complete! Final count: {total_generated}. Saved to: {dataset_file}")

if __name__ == "__main__":
    generate_samples(target_count=1500)
