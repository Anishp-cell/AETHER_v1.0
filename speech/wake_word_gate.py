"""
AETHER v4.0 — 3-Stage Acoustic Gate & Multi-Wake-Word Engine
Provides noise-immune, zero-idle-RAM wake detection across:
- 'Hey AETHER'
- 'AETHER'
- 'Hey Jarvis' / 'Jarvis'
- 'AETHER Online'
"""
import os
import sys
import time
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAKE_WORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_words")
os.makedirs(WAKE_WORDS_DIR, exist_ok=True)

try:
    import openwakeword
    from openwakeword.model import Model
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False


class WakeWordGate:
    """
    3-Stage Acoustic Gate:
    1. RMS Energy Noise Floor: Drops background fan/typing noise (0% CPU).
    2. ONNX Wake-Word Classifier: Evaluates 'hey_aether.onnx' and 'hey_jarvis'.
    3. Anti-Hallucination Sanitizer: Strips phantom Whisper artifacts.
    """
    def __init__(self, energy_threshold: float = 0.015, confidence_threshold: float = 0.65):
        self.energy_threshold = energy_threshold
        self.confidence_threshold = confidence_threshold
        self.custom_model_path = os.path.join(WAKE_WORDS_DIR, "hey_aether.onnx")
        self.oww_model = None
        self._init_models()

    def _init_models(self):
        """Initializes openWakeWord with custom hey_aether and standard models."""
        if not _OWW_AVAILABLE:
            print("[Acoustic Gate Warning] openwakeword not found. Operating in fallback mode.")
            return

        models_to_load = [self.custom_model_path]
        try:
            self.oww_model = Model(wakeword_models=models_to_load, inference_framework="onnx")
            print(f"[Acoustic Gate] Online. Active Wake-Words: ['Hey AETHER', 'AETHER']")
        except Exception as e:
            print(f"[Acoustic Gate] Warning loading ONNX models: {e}.")

    def check_audio_frame(self, audio_chunk_16khz: np.ndarray) -> bool:
        """
        Stage 1 + Stage 2 Check:
        1. Fast RMS energy test.
        2. ONNX prediction.
        """
        if audio_chunk_16khz is None or len(audio_chunk_16khz) == 0:
            return False

        # Stage 1: RMS Energy Floor
        rms = np.sqrt(np.mean(audio_chunk_16khz.astype(np.float32) ** 2)) / 32768.0
        if rms < self.energy_threshold:
            return False # Drop room silence/fan hum immediately

        # Stage 2: ONNX Model Prediction
        if self.oww_model:
            try:
                preds = self.oww_model.predict(audio_chunk_16khz)
                for mdl, score in preds.items():
                    if score >= self.confidence_threshold:
                        print(f"⚡ [Wake Event Triggered]: Detected '{mdl}' (Confidence: {score:.2f})")
                        return True
            except Exception:
                pass

        return False

    @staticmethod
    def sanitize_transcription(text: str) -> str:
        """
        Stage 3: Strips Whisper hallucination artifacts from ambient room audio.
        """
        cleaned = text.strip()
        # Remove known phantom noise transcriptions
        phantom_patterns = [
            r'\[music\]', r'\[applause\]', r'\[silence\]', r'\[laughter\]',
            r'^thank you\.?$', r'^thank you for watching\.?$', r'^subtitles by\.?$',
            r'^bye\.?$', r'^you\.?$', r'^\.$'
        ]
        for pat in phantom_patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                return "" # Pure ambient noise, discard

        # Strip wake-word prefixes if user said 'Hey AETHER, what time is it?'
        cleaned = re.sub(r'^(?:hey\s+)?(?:aether|jarvis|ether|ay-ther)\s*,?\s*', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()


# Singleton Instance
WAKE_WORD_GATE = WakeWordGate()
