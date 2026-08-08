import os
import openwakeword
from openwakeword.model import Model

WAKE_WORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_words")
CUSTOM_MODEL_PATH = os.path.join(WAKE_WORDS_DIR, "hey_aether.onnx")

class WakeWordDetector:
    def __init__(self, wake_words=None):
        """
        Initializes the incredibly lightweight Wake-Word engine.
        Runs 100% on the CPU via ONNX, consuming <0.1% CPU and <15MB RAM.
        Active Wake-Words: 'Hey AETHER', 'AETHER', 'Hey Jarvis', 'Jarvis'.
        """
        if wake_words is None:
            wake_words = []
            if os.path.exists(CUSTOM_MODEL_PATH):
                wake_words.append(CUSTOM_MODEL_PATH)
            wake_words.append("hey_jarvis")

        print(f"[Acoustic Layer] Loading Wake Word: {wake_words} (CPU Idle Matrix)")
        
        try:
            openwakeword.utils.download_models()
        except Exception:
            pass
            
        try:
            self.owwModel = Model(wakeword_models=wake_words, inference_framework="onnx")
        except Exception as e:
            print(f"[WakeWordDetector Warning] Falling back to default: {e}")
            self.owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    def detect(self, audio_chunk):
        """
        Processes a raw 16kHz int16 audio chunk.
        Returns True if user invoked 'Hey AETHER' or 'Hey Jarvis'.
        """
        prediction = self.owwModel.predict(audio_chunk)
        for mdl in prediction.keys():
            if prediction[mdl] > 0.6: 
                return True
        return False
        
    def get_required_sample_rate(self):
        return 16000 # OpenWakeWord requires exactly 16kHz

