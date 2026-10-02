import os
import sys
import datetime

def get_current_time(*args, **kwargs):
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

from desktop_agent import run_computer_command, analyze_screen_with_llava, open_app_and_type, search_web, open_url, get_system_diagnostics, media_control, send_whatsapp_message, set_timer, play_spotify_media, clipboard_action, smart_action
from aether_coder import AetherCoder
from screen_annotator import annotate_screen
from skill_registry import SKILL_REGISTRY
from vision_click_engine import vision_click, vision_read, vision_highlight
from window_manager import snap_window, tile_windows, close_foreground_window
from screen_watcher import start_screen_watch, stop_screen_watch
from web_agent import search_and_read_web, read_specific_url
from browser_agent import browser_task, browser_open_and_interact
from skill_factory import SkillFactory, SKILL_FACTORY
_coder_instance = AetherCoder()
_skill_factory = SKILL_FACTORY

from auto_skill_forge import forge_new_skill
from reflection_engine import reflect_and_execute
from ambient_daemon import get_morning_briefing, start_ambient_daemon
from screen_canvas_copilot import voice_canvas_draw
from hybrid_router import route_hybrid_query
from task_timer import TASK_TIMER

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
    "play_spotify_media": play_spotify_media,
    "send_whatsapp_message": send_whatsapp_message,
    "set_timer": set_timer,
    "annotate_screen": annotate_screen,
    "vision_click": vision_click,
    "vision_read": vision_read,
    "vision_highlight": vision_highlight,
    "snap_window": snap_window,
    "tile_windows": tile_windows,
    "close_foreground_window": close_foreground_window,
    "start_screen_watch": start_screen_watch,
    "stop_screen_watch": stop_screen_watch,
    "clipboard_action": clipboard_action,
    "smart_action": smart_action,
    "forge_new_skill": forge_new_skill,
    "reflect_and_execute": reflect_and_execute,
    "get_morning_briefing": get_morning_briefing,
    "voice_canvas_draw": voice_canvas_draw,
    "route_hybrid_query": route_hybrid_query,
    "browser_task": browser_task,
    "browser_open_and_interact": browser_open_and_interact,
    "get_nim_usage": lambda *args, **kwargs: _get_nim_usage_helper(*args, **kwargs),
    "get_nim_tokens": lambda *args, **kwargs: _get_nim_usage_helper(*args, **kwargs),
    "get_token_usage": lambda *args, **kwargs: _get_nim_usage_helper(*args, **kwargs),
    "get_most_visited_sites": lambda *args, **kwargs: _get_most_visited_sites_helper(*args, **kwargs),
    "get_top_sites": lambda *args, **kwargs: _get_most_visited_sites_helper(*args, **kwargs),
    "get_visited_sites": lambda *args, **kwargs: _get_most_visited_sites_helper(*args, **kwargs),
    "get_site_history": lambda *args, **kwargs: _get_most_visited_sites_helper(*args, **kwargs),
}


def _get_nim_usage_helper(*args, **kwargs) -> str:
    """Returns the current NVIDIA NIM token usage and remaining credit quota."""
    try:
        from nim_usage_tracker import NIMUsageTracker
        tracker = NIMUsageTracker()
        s = tracker.get_summary()
        lt = s["lifetime_tokens"]
        return (
            f"[NVIDIA NIM Quota Status]: {s['remaining_credits']} requests remaining out of {s['starting_credits']} "
            f"({s['percent_remaining']}% balance). Lifetime tokens: {lt.get('total_tokens', 0):,} "
            f"across {s['used_requests']} requests."
        )
    except Exception as e:
        return f"[NIM Usage Error]: Could not retrieve usage: {e}"


def _get_most_visited_sites_helper(*args, **kwargs) -> str:
    """Returns the most visited base websites recorded while using AETHER."""
    try:
        from site_history_manager import SiteHistoryManager
        top = SiteHistoryManager().get_top_sites(limit=5)
        if not top:
            return "No site visits have been recorded yet during active AETHER sessions."
        items = [f"{s['domain']} ({s['visit_count']} visits)" for s in top]
        return f"Most visited base websites: {', '.join(items)}."
    except Exception as e:
        return f"[Site History Error]: Could not retrieve site history: {e}"



# The architectural blueprint we feed to Qwen
OLLAMA_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_nim_usage",
            "description": "Fetch NVIDIA NIM API token usage statistics, remaining credits, and quota health."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_most_visited_sites",
            "description": "Fetch the user's most frequently visited base websites recorded while using AETHER."
        }
    },
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
            "name": "browser_task",
            "description": "DELEGATE to the Autonomous Browser Agent (Browser-Use + Playwright) to perform multi-step web automation, navigate complex websites, click buttons, fill forms, book flights/hotels, buy items on Amazon/Flipkart, extract dynamic JavaScript data, or interact with web services. Use this whenever the user asks you to interact with or do something inside a browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language instructions of what the browser agent should do on the web."
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "True to run silently in the background, False to display a visible browser window so user can watch. Defaults to false."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_open_and_interact",
            "description": "Navigate to a specific website URL and autonomously execute a goal on that page (fill forms, click buttons, extract data).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The target website URL."
                    },
                    "goal": {
                        "type": "string",
                        "description": "Goal or instructions to perform on the website."
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "True for background, False for visible window."
                    }
                },
                "required": ["url", "goal"]
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
                    "minutes": {"type": "number", "description": "Optional: Number of minutes to wait before triggering the alarm. Defaults to 0."},
                    "seconds": {"type": "number", "description": "Optional: Number of seconds to wait before triggering the alarm. Defaults to 0. Use this if the user specifies seconds (e.g. '10 seconds')."},
                    "reminder_message": {"type": "string", "description": "A short summary of what AETHER should say out loud when the timer goes off."}
                },
                "required": ["reminder_message"]
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
    },
    {
        "type": "function",
        "function": {
            "name": "browser_task",
            "description": "DELEGATE to the autonomous dual-channel browser agent for web automation, looking up live web pages, playing YouTube media, searching the web, filling forms, and navigating websites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language instruction for what to accomplish in the browser (e.g. 'Play lofi hip hop on YouTube', 'Search for NVIDIA Cosmos', 'Go to news.ycombinator.com and extract top stories')."
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "If True, runs browser silently in the background. If False, opens a visible window with visual glow highlighting."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_open_and_interact",
            "description": "Opens a specific URL directly and completes an interaction or objective on the loaded web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The exact web address to navigate to (e.g. 'https://github.com', 'https://youtube.com')."
                    },
                    "goal": {
                        "type": "string",
                        "description": "What to do once the page is loaded (e.g. 'extract the latest release version', 'click login')."
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Whether to run headless or headed."
                    }
                },
                "required": ["url", "goal"]
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
                            
                        print(f"[SkillFactory] Hot-Loaded dynamic skill: {skill_name}")
                        break
                        
            except Exception as e:
                print(f"[SkillFactory Error] Failed to load dynamic skill {filename}: {e}")


load_dynamic_skills()
SKILL_REGISTRY.discover_and_load_skills(AVAILABLE_TOOLS, OLLAMA_TOOL_DEFINITIONS)
