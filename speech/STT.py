import time
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
MODEL_SIZE = "small.en"      # "base.en" is faster, "small.en" is more accurate
THRESHOLD = 0.02             # RMS amplitude threshold (tweak if env is noisy)
SILENCE_TIMEOUT = 1.0        # Seconds of silence to consider the sentence "done"
PRE_BUFFER_DURATION = 0.5    # Seconds of audio to keep before trigger (catches "Hi")
SAMPLE_RATE = 16000
BLOCK_SIZE = 512             # Audio chunk size

# --- STATE ---
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Callback for sounddevice: pushes raw audio chunks to queue."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def main():
    # 1. Load Model
    print(f"Loading {MODEL_SIZE}...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print("Ready. Speak now.")

    # 2. Setup Audio Stream
    # We use a deque to hold the last X seconds of audio (Pre-buffer)
    # This ensures we don't cut off the start of the sentence.
    pre_buffer_len = int(PRE_BUFFER_DURATION * SAMPLE_RATE / BLOCK_SIZE)
    pre_buffer = deque(maxlen=pre_buffer_len)
    
    recording_buffer = []
    is_recording = False
    last_speech_time = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
                        channels=1, dtype="float32", callback=callback):
        while True:
            try:
                # Process audio chunk
                chunk = audio_queue.get(timeout=0.1)
                
                # Measure Energy (Simple VAD)
                volume = np.linalg.norm(chunk) / len(chunk)  # RMS-like but faster/simpler
                # Or regular RMS: np.sqrt(np.mean(chunk**2))
                # For speed/simplicity, max abs value is also fine: np.max(np.abs(chunk))
                # Let's use max amplitude for simpler thresholding:
                peak = np.max(np.abs(chunk))

                current_time = time.time()

                if peak > THRESHOLD:
                    last_speech_time = current_time
                    if not is_recording:
                        print(" [Recording] ...")
                        is_recording = True
                        # Dump pre-buffer into recording
                        recording_buffer.extend(pre_buffer)
                    
                # Store data
                if is_recording:
                    recording_buffer.append(chunk)

                    # Check for Silence
                    if current_time - last_speech_time > SILENCE_TIMEOUT:
                        print(" [Processing] ...")
                        is_recording = False
                        
                        # Process Audio
                        if len(recording_buffer) > 0:
                            audio_data = np.concatenate(recording_buffer).flatten().astype(np.float32)
                            
                            # Transcribe
                            segments, _ = model.transcribe(
                                audio_data, 
                                language="en",
                                beam_size=5,
                                best_of=5,
                                condition_on_previous_text=False # Prevents loops
                            )
                            
                            text = " ".join([s.text for s in segments]).strip()
                            if text:
                                print(f">>> {text}")
                            else:
                                print(f">>> (no speech detected)")
                            
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
