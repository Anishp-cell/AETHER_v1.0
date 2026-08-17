import os
import math
import wave
import struct
import threading
import sys

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

CHIME_ACTION_PATH = os.path.join(AUDIO_DIR, "chime_action.wav")
CHIME_SUCCESS_PATH = os.path.join(AUDIO_DIR, "chime_success.wav")

def generate_tactical_chime(file_path: str, freq1: float = 880.0, freq2: float = 1320.0, duration_ms: int = 140):
    """
    Generates a crisp, sub-150ms cyberpunk/tactical sine chime with exponential decay.
    100% pure Python standard library — zero external dependencies.
    """
    sample_rate = 22050
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1) # Mono
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = []
        for i in range(num_samples):
            t = i / sample_rate
            # Exponential volume decay curve
            envelope = math.exp(-t * 22.0)
            
            # Harmonic dual-sine blend
            sample_val = (
                0.6 * math.sin(2.0 * math.pi * freq1 * t) +
                0.4 * math.sin(2.0 * math.pi * freq2 * t)
            ) * envelope
            
            # Clamp to 16-bit integer range
            clamped = max(-32767, min(32767, int(sample_val * 32767 * 0.7)))
            frames.append(struct.pack('<h', clamped))
            
        wav_file.writeframes(b''.join(frames))

# Pre-generate chimes on import if not already created
if not os.path.exists(CHIME_ACTION_PATH):
    generate_tactical_chime(CHIME_ACTION_PATH, freq1=780.0, freq2=1170.0, duration_ms=130)

if not os.path.exists(CHIME_SUCCESS_PATH):
    generate_tactical_chime(CHIME_SUCCESS_PATH, freq1=920.0, freq2=1380.0, duration_ms=150)

def play_tactical_chime(chime_type: str = "action"):
    """
    Plays a sub-150ms tactical audio confirmation asynchronously.
    Bypasses LLM text generation and Piper/TTS for ultra-fast OS feedback.
    """
    def _player():
        try:
            target_file = CHIME_SUCCESS_PATH if chime_type == "success" else CHIME_ACTION_PATH
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(target_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                import subprocess
                subprocess.Popen(["aplay", "-q", target_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            pass # Non-critical fallback
            
    threading.Thread(target=_player, daemon=True).start()

if __name__ == "__main__":
    print("Testing tactical chime generation...")
    generate_tactical_chime(CHIME_ACTION_PATH)
    play_tactical_chime("action")
    print(f"Chime generated at: {CHIME_ACTION_PATH}")
