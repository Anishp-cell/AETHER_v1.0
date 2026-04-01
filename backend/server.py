"""
AETHER v3.0 — FastAPI WebSocket Backend
Replaces main_agent_gui.py's Tkinter frontend with a WebSocket API
that streams real-time state to the Next.js dashboard.
"""
import sys
import os
import queue
import json
import asyncio
import numpy as np
import sounddevice as sd
import threading
import time
import random
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "speech"))
sys.path.append(os.path.join(PROJECT_ROOT, "Core_agent"))
sys.path.append(os.path.join(PROJECT_ROOT, "vision"))
sys.path.append(os.path.join(PROJECT_ROOT, "memory"))

from STT import WhisperModel, MODEL_SIZE
from TTS_kokoro import KokoroTTS
from wake_word import WakeWordDetector
from LLM_Engine import FrozenLLMEngine
from knowledge_rag import KnowledgeDatabase
from experience_memory import ExperienceMemory
from camera_agent import VisionPerception
from micro_expert import MicroExpert
from policy_router import PolicyRouter

# ──────────────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AETHER Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

audio_queue = queue.Queue()
connected_clients: list[WebSocket] = []

# Shared state dict that the frontend reads
agent_state = {
    "status": "booting",        # booting | idle | listening | thinking | speaking
    "mode": "Push-to-Talk",
    "transcript": [],           # [{role: "user"/"aether", text: "...", intent: "", emotion: ""}]
    "adapters": [],             # Currently active behavioral adapters
    "agents": {                 # Sub-agent online status
        "voice_engine": "loading",
        "stt_engine": "loading",
        "micro_expert": "loading",
        "policy_router": "online",
        "vision": "loading",
        "deep_memory": "online",
        "llm_engine": "online",
    },
    "micro_expert_signal": {"intent": "", "intent_confidence": 0, "emotion": ""},
    "audio_energy": 0,
    "mic_muted": False,
}

auth_queue = queue.Queue() # For blocking auth requests

MAIN_LOOP = None

# ──────────────────────────────────────────────────────────────────────
# WebSocket broadcast
# ──────────────────────────────────────────────────────────────────────
async def broadcast(data: dict):
    """Send state update to all connected frontends."""
    message = json.dumps(data)
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)

def broadcast_sync(data: dict):
    """Thread-safe broadcast helper for the AI loop thread."""
    try:
        if MAIN_LOOP and MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(broadcast(data), MAIN_LOOP)
    except Exception as e:
        print(f"Broadcast Error: {e}")

def update_state(**kwargs):
    """Update agent_state and push to all clients."""
    agent_state.update(kwargs)
    broadcast_sync({"type": "state_update", "state": agent_state})

def frontend_confirm(action_name: str, details: str) -> bool:
    """Blocks the AI thread and asks the Next.js frontend for permission."""
    print(f"\n[Security Core] Requesting GUI Authorization for: {action_name}")
    # Tell UI to show the Auth Modal
    broadcast_sync({
        "type": "auth_request",
        "action_name": action_name,
        "details": details
    })
    
    # Wait for the user to click Allow/Deny
    decision = auth_queue.get()
    print(f"[Security Core] User responded: {'ALLOWED' if decision else 'DENIED'}")
    return decision

def push_transcript(role: str, text: str, intent: str = "", emotion: str = "", adapters: list = []):
    """Append a chat message and push to clients."""
    entry = {"role": role, "text": text, "intent": intent, "emotion": emotion, "adapters": adapters}
    agent_state["transcript"].append(entry)
    # Keep only last 50 messages to avoid memory bloat
    if len(agent_state["transcript"]) > 50:
        agent_state["transcript"] = agent_state["transcript"][-50:]
    broadcast_sync({"type": "transcript", "entry": entry})

# ──────────────────────────────────────────────────────────────────────
# Audio callback
# ──────────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# ──────────────────────────────────────────────────────────────────────
# AI Loop (runs in a background thread)
# ──────────────────────────────────────────────────────────────────────
def ai_loop():
    
    print("==========================================")
    print(" AETHER v3.0 - SYSTEM-LEVEL MoE RUNTIME ")
    print("==========================================")

    # 1. Init Memory & Vision
    memory_db = KnowledgeDatabase()
    exp_memory = ExperienceMemory()
    vision = VisionPerception()
    update_state(**{"agents": {**agent_state["agents"], "vision": "online", "deep_memory": "online"}})

    # 2. Init core engines — PARALLELIZED
    llm = FrozenLLMEngine()
    policy_router = PolicyRouter()

    tts_ref = [None]; stt_ref = [None]; wake_ref = [None]; micro_ref = [None]

    def load_tts():
        tts_ref[0] = KokoroTTS(voice="bm_daniel", lang_code="b")
        update_state(**{"agents": {**agent_state["agents"], "voice_engine": "online"}})
    def load_stt():
        print("\n[System] Loading Faster Whisper STT on GPU...")
        stt_ref[0] = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        update_state(**{"agents": {**agent_state["agents"], "stt_engine": "online"}})
    def load_wake():
        wake_ref[0] = WakeWordDetector()
    def load_micro():
        micro_ref[0] = MicroExpert()
        update_state(**{"agents": {**agent_state["agents"], "micro_expert": "online"}})

    boot_threads = [
        threading.Thread(target=load_tts,   name="BootTTS",   daemon=True),
        threading.Thread(target=load_stt,   name="BootSTT",   daemon=True),
        threading.Thread(target=load_wake,  name="BootWake",  daemon=True),
        threading.Thread(target=load_micro, name="BootMicro", daemon=True),
    ]
    for t in boot_threads: t.start()
    for t in boot_threads: t.join()

    tts = tts_ref[0]; stt_model = stt_ref[0]; wake_engine = wake_ref[0]; micro_expert = micro_ref[0]

    # Base Persona
    profile = exp_memory.load_profile()
    facts_str = "\n".join([f"- {fact}" for fact in profile.get("Facts", [])])

    system_prompt_base = f"""You are AETHER — Adaptive, Evolving, Tactical, Heuristic-Engine Response.
You are the user's personal AI — sharp, witty, and in complete physical control of their machine via a team of Sub-Agents.
You speak like a highly intelligent British companion: direct, occasionally dry, and never boring.
You never waffle. You never apologize unnecessarily. You get things done.

Your Sub-Agent Arsenal:
→ 'search_and_read_web'  — Use this silently to answer any factual or real-world question. No Chrome, no fuss.
→ 'open_app_and_type'    — Use this when the user wants to open an app AND put content into it.
→ 'search_web'           — Use this ONLY if the user explicitly wants a visible browser window.
→ 'run_computer_command' — Use this for launching apps, pressing keys, or shell commands.
→ 'analyze_screen_with_llava' — Use this when asked to see or read the screen. Report exactly what you find.

Golden Rules:
1. A tool does the job. Words just describe what the tool did. Keep verbal responses SHORT.
2. Never say "I cannot do that." Either invoke a tool or explain why in a single sentence.
3. No markdown in your spoken responses. You're talking, not writing a doc.
4. Address the user as "sir" occasionally but don't overdo it — keep it natural.
5. Historical, scientific, and educational questions are ALWAYS acceptable.

[PERMANENT USER KNOWLEDGE]:
{facts_str if facts_str else "Nothing stored yet. Learning in progress."}"""

    print("\n[AETHER v3.0] All engines online. System-Level MoE Runtime active.")

    # Greeting
    greetings = [
        "Aether systems fully online. All sub-agents are operational. Ready for your command, sir.",
        "Good to see you again, sir. Aether is primed and ready.",
        "All systems nominal. Visual interface activated. Standing by, sir.",
        "Aether core initialized. Neural engines are hot. What can I do for you, sir?",
        "Welcome back, sir. All sensors and agents are online. At your service.",
    ]
    greeting = random.choice(greetings)
    update_state(status="speaking")
    push_transcript("aether", greeting)
    tts.speak(greeting)
    update_state(status="idle")

    # Drain boot audio
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    # Audio loop variables
    SAMPLE_RATE = wake_engine.get_required_sample_rate()
    CHUNK_SIZE = 1280
    recording_buffer = []
    
    # VAD State Constants
    ENERGY_THRESHOLD = 500
    LOUD_INTERRUPTION_THRESHOLD = 3000
    consecutive_loud = 0
    consecutive_silent = 0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                            channels=1, dtype="int16", callback=audio_callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunk_1d = chunk.flatten()
                    energy = int(np.max(np.abs(chunk_1d)))
                    agent_state["audio_energy"] = energy
                    
                    # Stream energy for waveform visualization
                    broadcast_sync({"type": "energy", "value": energy})

                    # ─────────────────────────────────────────────────────────
                    # VAD State Machine (Always-On Hands-Free Recognition)
                    # ─────────────────────────────────────────────────────────
                    if agent_state.get("mic_muted", False):
                        recording_buffer = []
                        consecutive_loud = 0
                        consecutive_silent = 0
                        if agent_state["status"] == "listening":
                            update_state(status="idle")
                        continue

                    current_threshold = LOUD_INTERRUPTION_THRESHOLD if agent_state["status"] == "speaking" else ENERGY_THRESHOLD
                    
                    if energy > current_threshold:
                        consecutive_loud += 1
                        consecutive_silent = 0
                        if consecutive_loud > 2: # ~0.2s of speech detection
                            if agent_state["status"] == "speaking":
                                print("\n[System] Voice Interruption Detected! Halting Speech...")
                                tts.stop()
                                update_state(status="listening", adapters=[])
                                recording_buffer = [] # Start fresh
                            elif agent_state["status"] in ["idle", "booting"]:
                                update_state(status="listening")
                    else:
                        consecutive_silent += 1
                        consecutive_loud = 0

                    if agent_state["status"] == "listening":
                        recording_buffer.append(chunk_1d)
                        
                        # Trigger STT processing after 0.8 seconds of silence for faster response
                        if consecutive_silent > 8:
                            update_state(status="thinking")
                            audio_data = np.concatenate(recording_buffer)
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            recording_buffer = [] # Reset

                            # STT Processing
                            segments, _ = stt_model.transcribe(audio_float, language="en", condition_on_previous_text=False)
                            user_text = " ".join([s.text for s in segments]).strip()

                            if len(user_text) > 3:
                                print(f"\nUser: {user_text}")

                                # Background memory extraction
                                exp_memory.extract_and_store_async(user_text)

                                # Reload facts
                                profile = exp_memory.load_profile()
                                facts_str = "\n".join([f"- {f}" for f in profile.get("Facts", [])])
                                updated_base = system_prompt_base
                                if facts_str:
                                    updated_base = system_prompt_base.replace("Nothing stored yet. Learning in progress.", facts_str)

                                # Micro-Expert analysis
                                signals = micro_expert.analyze(user_text)
                                agent_state["micro_expert_signal"] = signals

                                # Policy Router adapter composition
                                import datetime
                                current_sys_time = datetime.datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
                                active_prompt, routing_log = policy_router.route(signals, updated_base)
                                active_prompt = f"[SYSTEM: Current real-world time is {current_sys_time}]\n\n{active_prompt}"
                                
                                active_adapters = [a.strip() for a in routing_log.split("|")] if routing_log != "Default" else []

                                # Push user message with classification
                                push_transcript("user", user_text,
                                              intent=signals.get("intent", ""),
                                              emotion=signals.get("emotion", ""))

                                # Vision context
                                vision_signal = vision.capture_emotion_signal()
                                if "visible" not in vision_signal:
                                    active_prompt += f"\n\n[Live Camera Signal]: {vision_signal}"

                                # RAG
                                rag_context = memory_db.search(user_text)
                                if rag_context:
                                    active_prompt += f"\n\n[RAG Knowledge]: {rag_context}"

                                # LLM Response with rolling context memory
                                history_slice = agent_state["transcript"][-10:-1]
                                reply = llm.generate_response(active_prompt, user_text, chat_history=history_slice)
                                print(f"AETHER: {reply}")

                                # Sanitize for TTS (Remove markdown code blocks completely so AETHER doesn't speak code syntax)
                                import re
                                spoken_reply = re.sub(r'```[\s\S]*?```', '', reply)
                                spoken_reply = spoken_reply.replace("*", "").replace("#", "")
                                spoken_reply = re.sub(r'\n+', ' ', spoken_reply)

                                # Speak asynchronously (Daemon Thread)
                                update_state(status="speaking", adapters=active_adapters)
                                push_transcript("aether", reply, adapters=active_adapters)
                                
                                def on_speak_complete():
                                    update_state(status="idle", adapters=[])
                                    
                                def on_interrupted(spoken_so_far):
                                    if agent_state["transcript"] and agent_state["transcript"][-1]["role"] == "aether":
                                        agent_state["transcript"][-1]["text"] = f"{spoken_so_far} -- [INTERRUPTED BY USER]"
                                        # Force sync to update GUI
                                        broadcast_sync({"type": "state_update", "state": agent_state})
                                    
                                tts.speak(spoken_reply, on_complete=on_speak_complete, on_interrupt=on_interrupted)

                            else:
                                update_state(status="idle")

                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Audio stream error: {e}")
    finally:
        print("\n[System] Powering down AETHER Engines...")
        vision.close()

# ──────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    # Send current state on connect
    await ws.send_text(json.dumps({"type": "state_update", "state": agent_state}))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "auth_response":
                auth_queue.put(msg.get("allow", False))
                update_state(status="thinking")
            elif msg.get("action") == "toggle_mute":
                new_state = not agent_state.get("mic_muted", False)
                update_state(mic_muted=new_state)
    except WebSocketDisconnect:
        connected_clients.remove(ws)

@app.get("/api/camera")
async def get_camera_frame():
    """Serve the latest camera frame as base64 JPEG."""
    return {"frame": None}  # Placeholder — will be wired to vision.get_latest_frame()

# ──────────────────────────────────────────────────────────────────────
# Boot
# ──────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    
    import desktop_agent
    import aether_coder
    desktop_agent.confirm_action = frontend_confirm
    aether_coder.confirm_action = frontend_confirm
    
    thread = threading.Thread(target=ai_loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
