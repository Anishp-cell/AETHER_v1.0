import time
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from STT import WhisperModel, MODEL_SIZE, THRESHOLD, SILENCE_TIMEOUT, PRE_BUFFER_DURATION, SAMPLE_RATE, BLOCK_SIZE, audio_queue, callback
from TTS import TextToSpeech

def main():
    print("Initializing AETHER Phase 1: Real-time Audio Loop...")
    
    # 1. Load Models
    print(f"Loading Faster Whisper ({MODEL_SIZE}) on GPU...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    
    tts = TextToSpeech()
    
    print("\n--- Audio System Ready ---")
    print("Speak into your microphone. Say something, and AETHER will echo it back.")
    print("Press Ctrl+C to stop.\n")

    # 2. Setup Audio Buffer
    pre_buffer_len = int(PRE_BUFFER_DURATION * SAMPLE_RATE / BLOCK_SIZE)
    pre_buffer = deque(maxlen=pre_buffer_len)
    
    recording_buffer = []
    is_recording = False
    last_speech_time = 0.0

    tts.speak("Audio loop initialized. I am listening.")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
                            channels=1, dtype="float32", callback=callback):
            while True:
                try:
                    # Fetch next audio chunk
                    chunk = audio_queue.get(timeout=0.1)
                    
                    # Measure voice activity simpler thresholding
                    peak = np.max(np.abs(chunk))
                    current_time = time.time()

                    if peak > THRESHOLD:
                        last_speech_time = current_time
                        if not is_recording:
                            print("\n[STT: Recording]...", end="", flush=True)
                            is_recording = True
                            recording_buffer.extend(pre_buffer)
                        
                    if is_recording:
                        recording_buffer.append(chunk)

                        # Check for End of Speech
                        if current_time - last_speech_time > SILENCE_TIMEOUT:
                            print(" [Done]")
                            is_recording = False
                            
                            if len(recording_buffer) > 0:
                                audio_data = np.concatenate(recording_buffer).flatten().astype(np.float32)
                                
                                # Process Transcribe
                                print("[STT: Transcribing]...", end="", flush=True)
                                segments, _ = model.transcribe(
                                    audio_data, 
                                    language="en",
                                    beam_size=5,
                                    condition_on_previous_text=False
                                )
                                
                                text = " ".join([s.text for s in segments]).strip()
                                
                                if text:
                                    print(f"\nUser: {text}")
                                    tts.speak(text)  # Echo back
                                else:
                                    print(" (No speech detected)")
                            
                            # Reset buffers
                            recording_buffer = []
                            pre_buffer.clear()
                            print("\nListening...")
                    else:
                        pre_buffer.append(chunk)

                except queue.Empty:
                    continue
                    
    except KeyboardInterrupt:
        print("\nStopping audio loop...")
        
if __name__ == "__main__":
    main()
