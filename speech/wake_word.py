import openwakeword
from openwakeword.model import Model
import numpy as np
import time

class WakeWordDetector:
    def __init__(self, wake_words=["hey_jarvis"]):
        """
        Initializes the incredibly lightweight Wake-Word engine.
        This runs 100% on the CPU via ONNX, consuming almost 0% power,
        freeing up the RTX 5050 and the Whisper model to completely sleep until called.
        """
        print(f"[Acoustic Layer] Loading Wake Word: {wake_words} (CPU Idle Matrix)")
        
        # openwakeword automatically caches public models internally
        try:
            openwakeword.utils.download_models()
        except:
            pass # Fails gracefully if already downloaded
            
        self.owwModel = Model(wakeword_models=wake_words, inference_framework="onnx")

    def detect(self, audio_chunk):
        """
        Processes a raw 16kHz int16 audio chunk.
        Returns True if the user successfully invoked the Wake Word.
        """
        # Ensure audio chunk is matching the shape openwakeword expects
        prediction = self.owwModel.predict(audio_chunk)
        
        for mdl in prediction.keys():
            # A score > 0.5 indicates a high confidence detection
            if prediction[mdl] > 0.6: 
                return True
        return False
        
    def get_required_sample_rate(self):
        return 16000 # OpenWakeWord requires exactly 16kHz
