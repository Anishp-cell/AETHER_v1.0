import time
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from faster_whisper import WhisperModel

# --- CONFIGURATION (Option 2: High-Speed base.en + Prompt Armor + VAD) ---
MODEL_SIZE = "base.en"        # Ultra-fast sub-450ms CPU execution
THRESHOLD = 0.025             # Hard RMS amplitude threshold to reject ambient room noise
SILENCE_TIMEOUT = 0.9         # Seconds of silence to consider the sentence "done"
PRE_BUFFER_DURATION = 0.4     # Catch start of rapid speech
SAMPLE_RATE = 16000
BLOCK_SIZE = 512              # Audio chunk size

# --- STATE ---
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Callback for sounddevice: pushes raw audio chunks to queue."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def main():
    # 1. Load Model with int8 CPU optimization
    print(f"Loading Faster-Whisper {MODEL_SIZE} (CPU int8 with Prompt Armor)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Ready. Speak now.")

    pre_buffer_len = int(PRE_BUFFER_DURATION * SAMPLE_RATE / BLOCK_SIZE)
    pre_buffer = deque(maxlen=pre_buffer_len)
    
    recording_buffer = []
    is_recording = False
    last_speech_time = 0.0

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
                        print(" [Recording] ...")
                        is_recording = True
                        recording_buffer.extend(pre_buffer)
                    
                if is_recording:
                    recording_buffer.append(chunk)

                    # Check for Silence
                    if current_time - last_speech_time > SILENCE_TIMEOUT:
                        print(" [Processing] ...")
                        is_recording = False
                        
                        if len(recording_buffer) > 0:
                            audio_data = np.concatenate(recording_buffer).flatten().astype(np.float32)
                            
                            # Transcribe with Option 2 Prompt Armor & VAD
                            segments, _ = model.transcribe(
                                audio_data, 
                                language="en",
                                beam_size=5,
                                initial_prompt="AETHER assistant desktop voice command:",
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=500),
                                condition_on_previous_text=False # Prevents looping
                            )
                            
                            raw_text = " ".join([s.text for s in segments]).strip()
                            
                            # Anti-Hallucination Regex Sanitizer
                            cleaned = raw_text
                            for p in [r'\[music\]', r'thank you for watching', r'subtitles by', r'^thank you\.?$', r'^bye\.?$']:
                                cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()

                            if cleaned:
                                print(f">>> {cleaned}")
                            else:
                                print(f">>> (silence / room noise filtered)")
                            
                            print("\nReady.")
                        
                        recording_buffer = []
                        pre_buffer.clear()
                else:
                    # If not recording, update pre-buffer Loop
                    pre_buffer.append(chunk)

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("\nStopping...")
                break

if __name__ == "__main__":
    main()
