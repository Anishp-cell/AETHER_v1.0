"""Test Aether-Orchestrator the way LLM_Engine.py actually calls it — plain chat, no Ollama tools API."""
import requests, json

TOOL_SCHEMA = """Available tools (output ONLY JSON, no other text):
- open_url: {"name":"open_url","arguments":{"url":"site_name_or_url"}}
- search_web: {"name":"search_web","arguments":{"query":"search_terms"}}
- run_computer_command: {"name":"run_computer_command","arguments":{"action_type":"open_app|press_key|type_text|shell_command","target":"value"}}
- send_whatsapp_message: {"name":"send_whatsapp_message","arguments":{"contact_name":"name","message":"text"}}
- get_system_diagnostics: {"name":"get_system_diagnostics","arguments":{}}
- media_control: {"name":"media_control","arguments":{"action":"play_pause|next_track|prev_track|mute|set_volume"}}
- set_timer: {"name":"set_timer","arguments":{"minutes":N,"reminder_message":"text"}}
- open_app_and_type: {"name":"open_app_and_type","arguments":{"app_name":"app","text_to_type":"content"}}"""

TEST_PROMPTS = [
    "open youtube",
    "search for lofi music",
    "check my battery",
    "send a whatsapp to my boss saying I will be late",
    "pause the music",
    "open notepad",
    "set a timer for 10 minutes to take a break",
]

for prompt in TEST_PROMPTS:
    print(f"\n{'='*60}")
    print(f"USER: {prompt}")
    
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "Aether-Orchestrator",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }).json()
    
    content = resp.get("message", {}).get("content", "ERROR")
    print(f"MODEL: {content[:300]}")
  