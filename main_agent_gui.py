import sys
import os
import queue
import numpy as np
import sounddevice as sd
import threading
import tkinter as tk
import time
import random

try:
    import keyboard
except ImportError:
    print("[Error] 'keyboard' module not installed. Please run: pip install keyboard")
    sys.exit(1)

sys.path.append(os.path.join(os.path.dirname(__file__), "speech"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Core_agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "vision"))
sys.path.append(os.path.join(os.path.dirname(__file__), "memory"))

from STT import WhisperModel, MODEL_SIZE
from TTS_kokoro import KokoroTTS
from wake_word import WakeWordDetector
from LLM_Engine import FrozenLLMEngine
from knowledge_rag import KnowledgeDatabase
from experience_memory import ExperienceMemory
from camera_agent import VisionPerception
from micro_expert import MicroExpert
from policy_router import PolicyRouter

from interface.gui import AetherGUI

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        pass # print(status)
    audio_queue.put(indata.copy())

class AetherApp:
    def __init__(self):
        # 1. UI Bootstrap
        self.root = tk.Tk()
        self.gui = AetherGUI(self.root, on_mode_change=self.on_mode_change, on_exit=self.on_exit)
        self.running = True
        
        # 2. System States
        self.mode = "Wake Word"
        self.is_awake = False
        self.is_speaking = False
        
        # 3. Push to Talk variables
        self.ptt_active = False # is spacebar being held?
        
        # Global key hooks
        keyboard.on_press_key('space', self.on_space_press)
        keyboard.on_release_key('space', self.on_space_release)
        
        # Start AI thread
        self.ai_thread = threading.Thread(target=self.ai_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()

    def on_mode_change(self, new_mode):
        print(f"\n[Mode Switched] -> {new_mode}")
        self.mode = new_mode

    def on_space_press(self, e):
        if self.mode == "Push-to-Talk" and not self.is_speaking:
            self.ptt_active = True
            self.is_awake = True

    def on_space_release(self, e):
        if self.mode == "Push-to-Talk":
            self.ptt_active = False

    def on_exit(self):
        print("Initiating full shutdown sequence...")
        self.running = False
        # Speak goodbye BEFORE killing mainloop (daemon thread dies instantly after)
        goodbyes = [
            "Goodbye, sir. Powering down all systems.",
            "Shutting down. Until next time, sir.",
            "All engines offline. Goodbye, sir.",
            "Aether signing off. Have a great day, sir.",
            "Going dark. See you soon, sir.",
        ]
        if hasattr(self, '_tts_ref') and self._tts_ref:
            try:
                self._tts_ref.speak(random.choice(goodbyes))
            except:
                pass

    def patched_tts_speak(self, tts_instance, text):
        """Wrapper around TTS to update GUI state while speaking"""
        self.gui.set_speaking_state(True)
        self.is_speaking = True
        
        # Actually speak (blocking)
        tts_instance.speak(text)
        
        self.gui.set_speaking_state(False)
        self.is_speaking = False

    def cam_updater(self, vision):
        """Poll the vision frame safely from Tkinter's main thread."""
        if self.running:
            frame = vision.get_latest_frame()
            if frame is not None:
                self.gui.set_camera_frame(frame)
            self.root.after(30, self.cam_updater, vision)

    def ai_loop(self):
        print("==========================================")
        print(" AETHER v3.0 - SYSTEM-LEVEL MoE RUNTIME ")
        print("==========================================")
        
        # 1. Init RAG Memory & Vision & Deep Memory
        memory_db = KnowledgeDatabase()
        exp_memory = ExperienceMemory()
        vision = VisionPerception()
        
        # Start GUI camera polling
        self.root.after(100, self.cam_updater, vision)
        
        # 2. Init core engines — PARALLELIZED for faster boot (Phase 7 Optimization)
        llm = FrozenLLMEngine()
        policy_router = PolicyRouter()
        
        tts_ref = [None]; stt_ref = [None]; wake_ref = [None]; micro_ref = [None]

        def load_tts():   tts_ref[0]   = KokoroTTS(voice="bm_daniel", lang_code="b")
        def load_stt():   
            print("\n[System] Loading Faster Whisper STT on GPU...")
            stt_ref[0] = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        def load_wake():  wake_ref[0]  = WakeWordDetector()
        def load_micro(): micro_ref[0] = MicroExpert()

        boot_threads = [
            threading.Thread(target=load_tts,   name="BootTTS",   daemon=True),
            threading.Thread(target=load_stt,   name="BootSTT",   daemon=True),
            threading.Thread(target=load_wake,  name="BootWake",  daemon=True),
            threading.Thread(target=load_micro, name="BootMicro", daemon=True),
        ]
        for t in boot_threads: t.start()
        for t in boot_threads: t.join()  # Wait for ALL engines to be ready

        tts          = tts_ref[0]
        stt_model    = stt_ref[0]
        wake_engine  = wake_ref[0]
        micro_expert = micro_ref[0]

        # Base Persona — the immutable foundation
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
5. Historical, scientific, and educational questions are ALWAYS acceptable. Discuss them factually and analytically without moral panic. A question about Hitler's military tactics is a history lesson, not an endorsement.

[PERMANENT USER KNOWLEDGE]:
{facts_str if facts_str else "Nothing stored yet. Learning in progress."}"""

        print("\n[AETHER v3.0] All engines online. System-Level MoE Runtime active.")

        
        # Random startup greeting
        greetings = [
            "Aether systems fully online. All sub-agents are operational. Ready for your command, sir.",
            "Good to see you again, sir. Aether is primed and ready.",
            "All systems nominal. Visual interface activated. Standing by, sir.",
            "Aether core initialized. Neural engines are hot. What can I do for you, sir?",
            "Welcome back, sir. All sensors and agents are online. At your service.",
            "Aether online. Desktop control, vision, and voice systems ready. Awaiting orders, sir.",
        ]
        self.patched_tts_speak(tts, random.choice(greetings))
        self._tts_ref = tts  # Store reference for goodbye

        # Phase 7: Drain any audio that piled up during the long boot sequence
        drained = 0
        while not audio_queue.empty():
            try: audio_queue.get_nowait(); drained += 1
            except: break
        if drained: print(f"[System] Audio queue flushed ({drained} stale chunks discarded).")

        # Audio Stream Parameters
        SAMPLE_RATE = wake_engine.get_required_sample_rate() # 16000
        CHUNK_SIZE = 1280 
        
        recording_buffer = []
        silence_frames = 0
        energy_threshold = 1000

        print("\n💤 [Awaiting Input]")
        
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, 
                                channels=1, dtype="int16", callback=audio_callback):
                while self.running:
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        chunk_1d = chunk.flatten()
                        
                        # Calculate audio energy for GUI visualization
                        energy = np.max(np.abs(chunk_1d))
                        # Only send energy to GUI when not speaking, to avoid visual feedback from speakers
                        if not self.is_speaking:
                            # Safely schedule GUI update
                            self.root.after(0, self.gui.set_mic_energy, energy)
                        
                        
                        # MODE: WAKE WORD
                        if self.mode == "Wake Word":
                            if not self.is_awake:
                                if wake_engine.detect(chunk_1d):
                                    self.is_awake = True
                                    print("\n🔔 [Wake Word Triggered!]")
                                    self.gui.set_listening_state(True)
                                    self.patched_tts_speak(tts, "Yes?") 
                                    print("🎙️ [Listening for command]...")
                            else:
                                recording_buffer.append(chunk_1d)
                                if energy > energy_threshold:
                                    silence_frames = 0
                                else:
                                    silence_frames += 1
                                    
                                if silence_frames > int((SAMPLE_RATE/CHUNK_SIZE) * 2.0):
                                    self._process_buffer(recording_buffer, SAMPLE_RATE, CHUNK_SIZE, stt_model, vision, memory_db, exp_memory, micro_expert, policy_router, llm, tts, system_prompt_base)
                                    # Reset
                                    self.is_awake = False
                                    recording_buffer = []
                                    silence_frames = 0
                                    self.root.after(0, self.gui.set_listening_state, False)
                                    print("\n💤 [Awaiting Wake Word]")
                        
                        # MODE: PUSH TO TALK
                        elif self.mode == "Push-to-Talk":
                            if self.ptt_active:
                                # Spacebar handles the recording phase
                                self.root.after(0, self.gui.set_listening_state, True)
                                recording_buffer.append(chunk_1d)
                            else:
                                if len(recording_buffer) > 0:
                                    # Spacebar released, process!
                                    self._process_buffer(recording_buffer, SAMPLE_RATE, CHUNK_SIZE, stt_model, vision, memory_db, exp_memory, micro_expert, policy_router, llm, tts, system_prompt_base)
                                    recording_buffer = []
                                    self.root.after(0, self.gui.set_listening_state, False)
                                    print("\n💤 [Awaiting Spacebar Press]")

                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Audio stream error: {e}")
        finally:
            print("\n[System] Powering down AETHER Engines...")
            vision.close()

    def _process_buffer(self, recording_buffer, SAMPLE_RATE, CHUNK_SIZE, stt_model, vision, memory_db, exp_memory, micro_expert, policy_router, llm, tts, base_prompt):
        if len(recording_buffer) < int(SAMPLE_RATE/CHUNK_SIZE):
            return
            
        audio_data_int16 = np.concatenate(recording_buffer)
        audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0 
        
        print("\n⚙️ [Processing audio]...", flush=True)
        self.root.after(0, self.gui.set_listening_state, False)
        
        segments, _ = stt_model.transcribe(audio_data_float32, language="en", condition_on_previous_text=False)
        user_text = " ".join([s.text for s in segments]).strip()
        
        if len(user_text) > 3:
            print(f"\nUser: {user_text}")
            
            # ── BACKGROUND MEMORY EXTRACTION ──
            exp_memory.extract_and_store_async(user_text)
            
            # Reload facts in case background thread just wrote new ones
            profile = exp_memory.load_profile()
            facts_str = "\n".join([f"- {fact}" for fact in profile.get("Facts", [])])
            updated_base = base_prompt
            if facts_str:
                updated_base = base_prompt.replace("Nothing stored yet. Learning in progress.", facts_str)
            
            # ── MICRO-EXPERT SIGNAL GENERATION (Claim 6) ──
            signals = micro_expert.analyze(user_text)
            
            # ── POLICY ROUTER ADAPTER COMPOSITION (Claims 1, 5, 7) ──
            active_prompt, routing_log = policy_router.route(signals, updated_base)
            
            # ── VISION CONTEXT INJECTION ──
            vision_signal = vision.capture_emotion_signal()
            if "visible" not in vision_signal:
                active_prompt += f"\n\n[Live Camera Signal]: {vision_signal}"
                
            # ── RAG KNOWLEDGE INJECTION ──
            rag_context = memory_db.search(user_text)
            if rag_context:
                active_prompt += f"\n\n[RAG Knowledge]: {rag_context}"
                
            reply = llm.generate_response(active_prompt, user_text)
            print(f"AETHER: {reply}")
            self.patched_tts_speak(tts, reply)

def main():
    app = AetherApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()
