import os
import json
import random
import requests
import time
import sys

# OpenRouter API Key (loaded from environment)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEYS = [OPENROUTER_KEY] if OPENROUTER_KEY else []


# Hardcoded Tool Definitions for portability
OLLAMA_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch the exact current time to answer queries about the time."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_smart_home",
            "description": "Physically control local smart devices like lights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "description": "The name of the device (lights, plug)"},
                    "action": {"type": "string", "description": "The state change (on or off)"}
                },
                "required": ["device", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_deepseek",
            "description": "MUST BE CALLED if the user asks any complex logic, hard math, deep reasoning, or physics problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The complex query to solve"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_app_and_type",
            "description": "MUST be used when the user wants to open an app AND write/type text into it. For example: 'open notepad and write hello', 'open notepad and type an essay'. This opens the app, waits for it to load, then pastes the text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "The application to open (e.g. 'notepad', 'wordpad')."},
                    "text_to_type": {"type": "string", "description": "The full text content to type/paste into the application."}
                },
                "required": ["app_name", "text_to_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "MUST be used when the user wants to navigate to a specific website like LinkedIn, YouTube, Instagram, Netflix, Gmail, or any URL. For example: 'go to LinkedIn', 'open YouTube', 'navigate to github.com'. Pass the site name or full URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The site name (e.g. 'linkedin') or full URL (e.g. 'https://www.linkedin.com') to open."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "MUST be used when the user wants to search for something on the web, Google, or Chrome. For example: 'search for paneer recipe', 'Google how to learn python', 'look up the weather'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g. 'paneer dal recipe', 'weather in Mumbai')."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_computer_command",
            "description": "DELEGATE to the Desktop Sub-Agent for simple PC actions: just opening an app (without typing), just pressing a key, just typing text into the currently focused window, or running a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["type_text", "press_key", "open_app", "shell_command", "mouse_click", "mouse_scroll", "keyboard_hotkey"], "description": "The category of action."},
                    "target": {"type": "string", "description": "The target to act upon: app name, key to press, shell command, text to type, 'x,y' for mouse click, y_amount (e.g. 500 or -500) for scroll, or 'ctrl,c' for hotkeys."}
                },
                "required": ["action_type", "target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_screen_with_llava",
            "description": "DELEGATE to the Vision Sub-Agent to read the user's screen. Use this when the user asks 'what am I looking at?' or 'read this text on screen' or 'what code is on my screen?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_query": {"type": "string", "description": "Specific instruction for the Vision agent."}
                },
                "required": ["task_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_read_web",
            "description": "DELEGATE to the AetherWeb Sub-Agent to silently scrape the internet to answer a real-world question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact fact-based search query to feed into the scraper."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_diagnostics",
            "description": "DELEGATE to the OS Macro tool when the user asks for a system status report, battery level, CPU, or RAM metrics. Takes no arguments."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "DELEGATE to the OS Macro tool when the user asks to play/pause media, skip a song, mute the computer, or set the master volume to an exact percentage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["play_pause", "next_track", "prev_track", "mute", "set_volume"], "description": "The exact media action to take."},
                    "value": {"type": "integer", "description": "Optional: Only used for 'set_volume' to define the 0-100 percentage."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "MANDATORY COMMAND: You MUST execute this tool IMMEDIATELY if the user asks you to text, msg, send a message to, or contact someone on WhatsApp. Do NOT just verbally say you will do it, you MUST output this JSON tool call so the backend can execute the script.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {"type": "string", "description": "The exact name of the WhatsApp contact."},
                    "message": {"type": "string", "description": "The text message content to send."}
                },
                "required": ["contact_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "DELEGATE to the OS Macro tool when the user wants an alarm, timer, or verbal reminder at a future time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "number", "description": "Number of minutes to wait before triggering the alarm. Convert hours to minutes if necessary (e.g. 1 hour = 60)."},
                    "reminder_message": {"type": "string", "description": "A short summary of what AETHER should say out loud when the timer goes off."}
                },
                "required": ["minutes", "reminder_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "teach_new_skill",
            "description": "MANDATORY: Trigger this tool if the user asks you to learn a new tool, write a permanent skill, create a new capability, or teach yourself how to do something permanently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "A snake_case name for the new python function (e.g., 'get_weather_data')."},
                    "description": {"type": "string", "description": "A short summary of what the new tool will do."},
                    "instruction": {"type": "string", "description": "The detailed prompt/instruction for the LLM to write the python code."}
                },
                "required": ["skill_name", "description", "instruction"]
            }
        }
    }
]

# OpenRouter configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_PROMPT = """You are a master dataset generator for a Reinforcement Learning AI pipeline. 
Your job is to generate synthetic training pairs mapping a human's spoken command to the EXACT JSON tool sequence required to achieve it.

**Rules for Generation**:
1. You must read the provided JSON Tool Definitions to understand what tools are available.
2. The user will specify a random scenario or theme.
3. You must generate 5 distinct pairs for that theme. 
4. DO NOT use specific real names (like "Manju" or "John"). Use generic terms like `<CONTACT_NAME>`, "my boss", "my friend", or just an arbitrary placeholder name.
5. The output MUST be a strict JSON object with a single key called "pairs", which contains an array of 5 objects. Each object must have a `user_prompt` string and an `expected_json` string.
6. For the `teach_new_skill` tool, ensure the `instruction` argument contains a very clear, detailed prompt for generating the python code, and `description` describes the tool succinctly.
7. Use diverse language, typos, slang, and complex compound instructions. Include examples where users correct themselves mid-sentence.

**Format Example**:
{
  "pairs": [
    {
      "user_prompt": "Send a WhatsApp message to my manager saying I will be late.",
      "expected_json": "[{\"name\": \"send_whatsapp_message\", \"arguments\": {\"contact_name\": \"my manager\", \"message\": \"I will be late.\"}}]"
    }
  ]
}

**Tool Definitions**:
""" + json.dumps(OLLAMA_TOOL_DEFINITIONS, indent=2)

SCENARIOS = [
    "Single basic desktop app manipulation (e.g. open calculator, notepad).",
    "Web navigation to single popular URLs (e.g. Youtube, Netflix).",
    "System diagnostics queries (battery, CPU, RAM).",
    "Sending messages via WhatsApp to generic contacts.",
    "Web search followed by typing into an application (compound task).",
    "Complex shell commands execution.",
    "Mouse clicking or scrolling based on user requesting navigation UI interaction.",
    "Keyboard hotkey pressing.",
    "Time checking and timer setting.",
    "Routing very complex physics or math to DeepSeek.",
    "Analyzing the screen because the user is confused about an error.",
    "Playing, pausing, or skipping media.",
    "Smart home light toggling.",
    "Python scripting generation and running.",
    "Compound task: Check battery, send whatsapp, and open a URL.",
    "Silent background web search for a specific fact.",
    "User asking the system to learn a new permanent skill to fetch cryptocurrency prices.",
    "User telling the AI to write a permanent tool that scrapes weather data.",
    "User asking Aether to teach itself how to parse a PDF file permanently.",
    "User asking the agent to forge a new capability for controlling a specific IoT device.",
    "User complaining about a missing feature and demanding the AI to code a new permanent skill for it."
]

def generate_samples(batch_size=5):
    dataset_file = os.path.join(os.path.dirname(__file__), "aether_orchestrator_dataset.jsonl")
    
    # Check if file exists to resume, otherwise create it
    if not os.path.exists(dataset_file):
        with open(dataset_file, 'w', encoding='utf-8') as f:
            pass
    
    # Count existing pairs to resume properly
    with open(dataset_file, 'r', encoding='utf-8') as f:
        total_generated = sum(1 for _ in f)
    
    key_idx = 0
    target_count = 1500
    
    if total_generated > 0:
        print(f"Resuming synthesis from {total_generated}/{target_count} pairs...")
    else:
        print(f"Starting Aether Orchestrator Dataset Synthesis (Target: {target_count} pairs)...")
    
    with open(dataset_file, 'a', encoding='utf-8') as f:
        while total_generated < target_count:
            scenario = random.choice(SCENARIOS)
            prompt = f"Generate {batch_size} varied command pairs for the scenario: {scenario}. Make them completely different from each other in phrasing. Some should be casual ('hey can you just...'), some direct ('do this'), some with typos or slang."
            
            target_key = GROQ_API_KEYS[key_idx].strip()
            print(f"Using key {key_idx} (starts with {target_key[:10]}...)")
            
            headers = {
                "Authorization": f"Bearer {target_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/aether-orchestrator", 
                "X-Title": "Aether Dataset Generator"
            }
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8,
                "response_format": {"type": "json_object"}
            }
            
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 429: # Rate limit
                    print(f"Rate limit hit on key {key_idx}. Sleeping...")
                    key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                    time.sleep(5) # Give it a bit more breathing room
                    continue
                
                if response.status_code != 200:
                    print(f"API Error {response.status_code}: {response.text}")
                    key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                    time.sleep(2)
                    continue
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and len(parsed.keys()) == 1:
                        pairs = list(parsed.values())[0]
                    else:
                        pairs = parsed
                        
                    if not isinstance(pairs, list):
                        raise ValueError("Output is not a list")
                        
                    for pair in pairs:
                        expected = pair.get('expected_json', '')
                        user_prompt = pair.get('user_prompt', '')
                        
                        if not expected or not user_prompt:
                            continue
                        
                        # CRITICAL FIX: Assistant output is RAW JSON only.
                        # No markdown fences. No explanation. Just the array.
                        finetune_row = {
                            "messages": [
                                {
                                    "role": "system", 
                                    "content": "You are a tool-calling orchestrator. Output ONLY a raw JSON array of tool calls. No text, no markdown, no explanation."
                                },
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": expected}
                            ]
                        }
                        f.write(json.dumps(finetune_row) + "\n")
                        total_generated += 1
                        
                    print(f"Synthesized {total_generated}/{target_count} pairs...")
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse generation: {e}")
                    
            except Exception as e:
                print(f"API Error: {e}")
                key_idx = (key_idx + 1) % len(GROQ_API_KEYS)
                time.sleep(2)

    print(f"\nDataset generated at: {dataset_file}")
    print(f"Total examples: {total_generated}")

if __name__ == "__main__":
    generate_samples(batch_size=5)

