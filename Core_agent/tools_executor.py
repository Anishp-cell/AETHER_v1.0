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

from desktop_agent import run_computer_command, analyze_screen_with_llava, open_app_and_type, search_web
from aether_coder import AetherCoder
import sys
import os

# Ensure web_agent is reachable
sys.path.append(os.path.dirname(__file__))
from web_agent import search_and_read_web, read_specific_url

_coder_instance = AetherCoder()

# The mapping dictionary that Ollama's tool call JSON directly hooks into
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "handle_smart_home": handle_smart_home,
    "route_to_deepseek": route_to_deepseek,
    "run_computer_command": run_computer_command,
    "analyze_screen_with_llava": analyze_screen_with_llava,
    "open_app_and_type": open_app_and_type,
    "search_web": search_web,
    "search_and_read_web": search_and_read_web,
    "read_specific_url": read_specific_url,
    "write_and_run_script": _coder_instance.write_and_run_script
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
                    "action_type": {"type": "string", "enum": ["type_text", "press_key", "open_app", "shell_command"], "description": "The category of action."},
                    "target": {"type": "string", "description": "The exact string to type, the key to press (e.g. 'enter'), the app name (e.g. 'notepad'), or the shell command."}
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
    }
]
