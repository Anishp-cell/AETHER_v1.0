import sys
import os
import queue
import numpy as np
import sounddevice as sd

sys.path.append(os.path.join(os.path.dirname(__file__), "speech"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Core_agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "vision"))
sys.path.append(os.path.join(os.path.dirname(__file__), "memory"))

from STT import WhisperModel, MODEL_SIZE
from TTS_piper import PiperTTS
from wake_word import WakeWordDetector
from LLM_Engine import FrozenLLMEngine
from knowledge_rag import KnowledgeDatabase
from camera_agent import VisionPerception

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def main():
    print("==========================================")
    print(" AETHER v2.0 - FULLY AUTONOMOUS LOOP ")
    print("==========================================")
    
    # 1. Init RAG Memory & Vision
    memory_db = KnowledgeDatabase()
    vision = VisionPerception()
    
    # 2. Init Dual-LLM & Piper & STT & Wake Word
    llm = FrozenLLMEngine()
    tts = PiperTTS()
    wake_engine = WakeWordDetector()
    
    print("\n[System] Loading Faster Whisper STT on GPU...")
    stt_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    
    system_prompt_base = "You are AETHER, a brilliant local agent. Answer concisely. Do not use markdown like asterisks."

    print("\n[AETHER 2.0] All engines loaded. Awaiting Wake Word: 'Hey Jarvis'.")
    tts.speak("Aether systems fully upgraded and online.")

    # Audio Stream Parameters
    SAMPLE_RATE = wake_engine.get_required_sample_rate() # 16000
    CHUNK_SIZE = 1280 
    
    is_awake = False
    recording_buffer = []
    silence_frames = 0
    energy_threshold = 1000 # Adjust if too sensitive 

    print("\n💤 [Sleeping... 0% VRAM / 0% GPU Usage]")
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, 
                            channels=1, dtype="int16", callback=audio_callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunk_1d = chunk.flatten()
                    
                    if not is_awake:
                        # Feed int16 audio softly to Wake Word Model on CPU
                        if wake_engine.detect(chunk_1d):
                            is_awake = True
                            print("\n🔔 [Wake Word Triggered!] Booting matrix...")
                            tts.speak("Yes?") # Play chime or voice
                            print("🎙️ [Listening for command]...", end="", flush=True)
                    else:
                        # AETHER is awake! Recording into the buffer for Whisper STT
                        recording_buffer.append(chunk_1d)
                        
                        # Calculate basic noise floor to detect when user stops speaking
                        energy = np.max(np.abs(chunk_1d))
                        if energy > energy_threshold:
                            silence_frames = 0
                            print(".", end="", flush=True)
                        else:
                            silence_frames += 1
                            
                        # If silence hits ~2 seconds, cut off the recording and send it
                        if silence_frames > int((SAMPLE_RATE/CHUNK_SIZE) * 2.0):
                            if len(recording_buffer) > int(SAMPLE_RATE/CHUNK_SIZE): # Drop glitches
                                audio_data_int16 = np.concatenate(recording_buffer)
                                audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0 
                                
                                print("\n⚙️ [Processing text]...", end="", flush=True)
                                segments, _ = stt_model.transcribe(audio_data_float32, language="en", condition_on_previous_text=False)
                                user_text = " ".join([s.text for s in segments]).strip()
                                
                                if len(user_text) > 3:
                                    print(f"\nUser: {user_text}")
                                    
                                    # Formulate unified context prompt (RAG Memory + Actionable Vision)
                                    active_prompt = system_prompt_base
                                    vision_signal = vision.capture_emotion_signal()
                                    if "visible" not in vision_signal:
                                        print(f"[Vision Active]: {vision_signal}")
                                        active_prompt += f"\n\n[Live Camera Sentiment]: {vision_signal}"
                                        
                                    rag_context = memory_db.search(user_text)
                                    if rag_context:
                                        active_prompt += f"\n\n[Historical RAG Memory]: {rag_context}"
                                        
                                    # Hit the Router (Qwen) which determines if tools or DeepSeek are needed!
                                    reply = llm.generate_response(active_prompt, user_text)
                                    print(f"AETHER: {reply}")
                                    tts.speak(reply)
                            
                            # Return to zero-power slumber
                            is_awake = False
                            recording_buffer = []
                            silence_frames = 0
                            print("\n💤 [Sleeping... 0% VRAM / 0% GPU Usage]")

                except queue.Empty:
                    continue
                    
    except KeyboardInterrupt:
        print("\n[System] Powering down AETHER Engines...")
        vision.close()
            
if __name__ == "__main__":
    main()
