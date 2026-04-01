import sys
import os
import time
import queue
import numpy as np
import sounddevice as sd
from collections import deque

# Ensure python can find our modules properly from the root directory
sys.path.append(os.path.join(os.path.dirname(__file__), "speech"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Core_agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "memory"))

from STT import WhisperModel, MODEL_SIZE, THRESHOLD, SILENCE_TIMEOUT, PRE_BUFFER_DURATION, SAMPLE_RATE, BLOCK_SIZE, audio_queue, callback
from TTS import TextToSpeech
from LLM_Engine import FrozenLLMEngine
from knowledge_rag import KnowledgeDatabase

def main():
    print("======================================")
    print(" AETHER v1.0 - LIVE INTELLIGENCE LOOP ")
    print("======================================")
    
    # Init Backend RAG Memory
    memory_db = KnowledgeDatabase()
    
    # Init Base LLM via Ollama
    llm = FrozenLLMEngine("qwen2.5:1.5b")
    # Tell the LLM not to use markdown since it will be spoken out loud.
    system_prompt = "You are AETHER, a brilliant, concise, and highly effective local AI agent. Provide clear, short answers. Do not use asterisks, emojis, or heavy formatting because your response will be read aloud."

    # 2. Init Perception
    print(f"\n[System] Loading Faster Whisper STT ({MODEL_SIZE}) on GPU...")
    stt_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    tts = TextToSpeech()
    
    print("\n--- All Systems Online ---")
    tts.speak("Aether systems interconnected. I am listening.")

    pre_buffer_len = int(PRE_BUFFER_DURATION * SAMPLE_RATE / BLOCK_SIZE)
    pre_buffer = deque(maxlen=pre_buffer_len)
    
    recording_buffer = []
    is_recording = False
    last_speech_time = 0.0
    start_listen_time = 0.0

    print("\nListening...")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
                            channels=1, dtype="float32", callback=callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    peak = np.max(np.abs(chunk))
                    current_time = time.time()

                    if peak > THRESHOLD:
                        last_speech_time = current_time
                        if not is_recording:
                            start_listen_time = current_time
                            print("\n[User Speaking]...", end="", flush=True)
                            is_recording = True
                            recording_buffer.extend(pre_buffer)
                        
                    if is_recording:
                        recording_buffer.append(chunk)

                        # End of Speech detected
                        if current_time - last_speech_time > SILENCE_TIMEOUT:
                            listen_duration = current_time - start_listen_time
                            is_recording = False
                            
                            if len(recording_buffer) > 0:
                                audio_data = np.concatenate(recording_buffer).flatten().astype(np.float32)
                                
                                # --- 1. STT Phase ---
                                t0_stt = time.time()
                                segments, _ = stt_model.transcribe(
                                    audio_data, 
                                    language="en",
                                    beam_size=5,
                                    condition_on_previous_text=False
                                )
                                user_text = " ".join([s.text for s in segments]).strip()
                                t1_stt = time.time()
                                
                                stt_time = t1_stt - t0_stt
                                
                                if user_text:
                                    print(f"\n[Heard in {listen_duration:.2f}s | Transcribed in {stt_time:.2f}s] User: {user_text}")
                                    
                                    # Very basic filter to avoid whisper hallucinating silence
                                    if len(user_text.strip()) > 3:
                                        
                                        # --- 2. LLM Phase (With RAG) ---
                                        t0_llm = time.time()
                                        print("[Thinking]...", end="", flush=True)
                                        
                                        # Search local memory for relevant context
                                        context = memory_db.search(user_text)
                                        
                                        active_prompt = system_prompt
                                        if context:
                                            print("\n[RAG: Recalled relevant memory]")
                                            active_prompt += f"\n\nHighly precise background context you must use to answer if relevant: {context}"
                                        
                                        reply = llm.generate_response(active_prompt, user_text)
                                        t1_llm = time.time()
                                        llm_time = t1_llm - t0_llm
                                        print(f"\n[Answered in {llm_time:.2f}s] AETHER: {reply}")
                                        
                                        # --- 3. TTS Phase ---
                                        tts.speak(reply)  # Technically PyTTSx3 is blocking, so time is just length of speech.
                                        
                                else:
                                    print("\n(No coherent speech detected)")
                            
                            # Clean up and restart loop
                            recording_buffer = []
                            pre_buffer.clear()
                            print("\nListening...")
                    else:
                        pre_buffer.append(chunk)

                except queue.Empty:
                    continue
                    
    except KeyboardInterrupt:
        print("\n\n[System] Shutting down AETHER...")
        
if __name__ == "__main__":
    main()
