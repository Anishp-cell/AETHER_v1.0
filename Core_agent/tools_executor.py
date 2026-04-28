import datetime

def get_current_time():
    """Returns the current precise time for daily alarms or scheduling."""
    return datetime.datetime.now().strftime("%I:%M %p")

def handle_smart_home(device="lights", action="on"):
    """
    Physical script to trigger IoT devices (like python-kasa in A.D.A).
    """
    return f"Successfully turned {action} the {device} based on your command."

def route_to_deepseek(query):
    """
    The Dual-Model trigger. 
    If Qwen2.5 classifies the query as "complex reasoning", it structurally calls this function.
    """
    return "[DEEPSEEK_ROUTING_ACTIVATED]" 

from desktop_agent import run_computer_command, analyze_screen_with_llava, open_app_and_type, search_web, open_url, get_system_diagnostics, media_control, send_whatsapp_message, set_timer
from aether_coder import AetherCoder
import sys
import os

# Ensure web_agent is reachable
sys.path.append(os.path.dirname(__file__))
from web_agent import search_and_read_web, read_specific_url

_coder_instance = AetherCoder()

from skill_factory import SkillFactory
_skill_factory = SkillFactory()

# The mapping dictionary that Ollama's tool call JSON directly hooks into
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "handle_smart_home": handle_smart_home,
    "route_to_deepseek": route_to_deepseek,
    "run_computer_command": run_computer_command,
    "analyze_screen_with_llava": analyze_screen_with_llava,
    "open_app_and_type": open_app_and_type,
    "search_web": search_web,
    "open_url": open_url,
    "search_and_read_web": search_and_read_web,
    "read_specific_url": read_specific_url,
    "write_and_run_script": _coder_instance.write_and_run_script,
    "teach_new_skill": _skill_factory.teach_new_skill,
    "get_system_diagnostics": get_system_diagnostics,
    "media_control": media_control,
    "send_whatsapp_message": send_whatsapp_message,
    "set_timer": set_timer
}


# The architectural blueprint we feed to Qwen
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
                    "task_query": {"type": "string", "description": "Specific instruction for the Vision agent (e.g. 'Read the python error' or 'Describe what is on screen')."}
                },
                "required": ["task_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_read_web",
            "description": "DELEGATE to the AetherWeb Sub-Agent to silently scrape the internet to answer a real-world question (e.g. 'Who is the president of France?', 'Summary of the Matrix movie', 'What is the capital of Japan?'). Do NOT use this if the user says 'open Chrome' or 'search for on Google', use search_web instead.",
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
            "name": "read_specific_url",
            "description": "DELEGATE to the AetherWeb Sub-Agent when the user explicitly provides a URL link to read (e.g. 'Read this URL to me https://example.com' or 'Summarize this webpage...').",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The HTTP URL to read and fetch."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_and_run_script",
            "description": "DELEGATE to the AetherCoder Sub-Agent when the user asks you to write, save, and execute a python script or write a program to do something recursively/automagically on their local file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "The clear natural language instruction detailing what the script must accomplish. e.g. 'Write a python script to rename all images in my downloads folder'"}
                },
                "required": ["instruction"]
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
            "description": "MANDATORY COMMAND: Use this tool ONLY when the user EXPLICITLY asks you to 'write a permanent new skill', 'learn a new tool', or 'permanently add a new capability'. Do NOT use for temporary scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "A short, snake_case python function name for the new skill (e.g. 'get_bitcoin_price')."},
                    "description": {"type": "string", "description": "A short description of what the skill does."},
                    "instruction": {"type": "string", "description": "The detailed instructions of how the code should be written."}
                },
                "required": ["skill_name", "description", "instruction"]
            }
        }
    }
]

import ast
import importlib.util

def load_dynamic_skills():
    """Scans the skills/ directory, parses Python files, and hot-loads them into the runtime."""
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    if not os.path.exists(skills_dir):
        return
        
    sys.path.insert(0, skills_dir)
    
    for filename in os.listdir(skills_dir):
        if filename.endswith(".py") and not filename.startswith("temp_"):
            skill_name = filename[:-3]
            filepath = os.path.join(skills_dir, filename)
            
            try:
                # Parse AST to get docstring and parameters
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == skill_name:
                        docstring = ast.get_docstring(node) or f"Dynamically loaded skill: {skill_name}"
                        
                        # Extract arguments for JSON Schema
                        properties = {}
                        required = []
                        for arg in node.args.args:
                            arg_name = arg.arg
                            properties[arg_name] = {"type": "string", "description": f"Argument {arg_name}"}
                            required.append(arg_name)
                            
                        # Build OLLAMA schema
                        schema = {
                            "type": "function",
                            "function": {
                                "name": skill_name,
                                "description": docstring,
                            }
                        }
                        if properties:
                            schema["function"]["parameters"] = {
                                "type": "object",
                                "properties": properties,
                                "required": required
                            }
                            
                        # Load module and inject
                        spec = importlib.util.spec_from_file_location(skill_name, filepath)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        func = getattr(module, skill_name)
                        
                        # Inject
                        AVAILABLE_TOOLS[skill_name] = func
                        
                        # Check if already in definitions to prevent duplicates on hot-reload if not cleared
                        if not any(t["function"]["name"] == skill_name for t in OLLAMA_TOOL_DEFINITIONS):
                            OLLAMA_TOOL_DEFINITIONS.append(schema)
                            
                        print(f"[SkillFactory] ⚡ Hot-Loaded dynamic skill: {skill_name}")
                        break
                        
            except Exception as e:
                print(f"[SkillFactory Error] Failed to load dynamic skill {filename}: {e}")

load_dynamic_skills()
