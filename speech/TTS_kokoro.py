import os
from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import torch
import warnings
import threading
import time

# Suppress some noisy FutureWarnings from torch/transformers
warnings.filterwarnings("ignore")

class KokoroTTS:
    def __init__(self, voice="bf_emma", lang_code="b"):
        """
        Initializes the Kokoro-82M TTS Engine on CPU to bypass CuDNN errors.
        """
        print(f"[TTS Layer] Initializing Kokoro-82M Voice Engine on CPU (Voice: {voice})...")
        
        # We temporarily blind torch to CUDA so Kokoro initializes strictly on CPU.
        # This prevents the Windows `cudnnGetLibConfig. Error code 127` crash,
        # but leaves the GPU intact for the Whisper STT engine!
        original_cuda_check = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        try:
            # Load the pipeline (Will download model/voice weights on first boot from HuggingFace)
            self.pipeline = KPipeline(lang_code=lang_code)
        finally:
            # Always restore CUDA availability for STT
            torch.cuda.is_available = original_cuda_check
        
        self.voice = voice
        self.sample_rate = 24000
        self.is_interrupted = False
        self.speak_thread = None

    def stop(self):
        """Immediately halts all active speech and flushes the generation pipeline."""
        self.is_interrupted = True

    def speak(self, text, on_complete=None, on_interrupt=None):
        """Asynchronous streaming playback with interruptible polling."""
        if self.speak_thread and self.speak_thread.is_alive():
            self.stop()
            self.speak_thread.join(timeout=1.0)
            
        def _speak_task():
            if not text:
                if on_complete: on_complete()
                return
                
            self.is_interrupted = False
            spoken_log = ""
            print("[Speaking]...", end="", flush=True)
            # Split by punctuation rather than newline so it generates sentence by sentence
            generator = self.pipeline(text, voice=self.voice, speed=1.0, split_pattern=r'(?<=[.!?])\s+')
            
            try:
                for i, (graphemes, phonemes, audio) in enumerate(generator):
                    if self.is_interrupted:
                        break
                    if audio is not None:
                        sd.play(audio, self.sample_rate)
                        
                        chunk_completed = True
                        # Poll stream actively so we can kill it mid-chunk
                        while getattr(sd.get_stream(), 'active', False):
                            if self.is_interrupted:
                                sd.stop()
                                chunk_completed = False
                                break
                            time.sleep(0.01)
                            
                        if chunk_completed:
                            spoken_log += graphemes + " "
            except Exception as e:
                print(f"\n[TTS Engine Error] Failed to generate audio: {e}")
            finally:
                if self.is_interrupted:
                    print(" [Interrupted]")
                    if on_interrupt:
                        on_interrupt(spoken_log.strip())
                else:
                    print(" [Done]")
                    if on_complete:
                        on_complete()

        self.speak_thread = threading.Thread(target=_speak_task, daemon=True)
        self.speak_thread.start()

if __name__ == "__main__":
    # Test script
    tts = KokoroTTS(voice="bm_daniel", lang_code="b") # bm_daniel is the british male daniel voice
    tts.speak("Hello sir, I am Aether. My voice engine has been successfully upgraded to Kokoro.")
