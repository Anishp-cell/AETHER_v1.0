import os
import subprocess
import time
try:
    import winsound
except ImportError:
    winsound = None

class PiperTTS:
    def __init__(self, model_path="models/voice/en_GB-northern_english_male-medium.onnx"):
        print("[TTS Layer] Initializing Piper ONNX Voice Engine (CPU Optimized)...")
        # Ensure we construct the absolute path correctly
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(base_dir, model_path)
        self.use_piper = True
        
        if not os.path.exists(self.model_path):
            print(f"[TTS Layer] Warning: Piper model not found at {self.model_path}. Defaulting to Windows Native Voice.")
            self.use_piper = False
            
        # Fallback Engine
        self._init_fallback()

    def _init_fallback(self):
        try:
            import pyttsx3
            self.fallback_engine = pyttsx3.init()
            self.fallback_engine.setProperty('rate', 160)
            voices = self.fallback_engine.getProperty('voices')
            if len(voices) > 1:
                self.fallback_engine.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"[TTS Layer] Fallback engine failed to init: {e}")
            self.fallback_engine = None

    def speak(self, text: str):
        if not text.strip():
            return
            
        print("[Speaking]...", end="", flush=True)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        output_file = os.path.join(base_dir, "temp_speech.wav")
        
        success = False
        
        if self.use_piper:
            try:
                piper_exe = "piper"
                
                # Spawn piper and output to temp_speech.wav
                process = subprocess.Popen(
                    [piper_exe, "-m", self.model_path, "-f", output_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                process.communicate(input=text.encode('utf-8'))
                
                # If piper worked, play it!
                if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                    if winsound:
                        # Play synchronously using winsound (bypasses sounddevice driver conflicts)
                        winsound.PlaySound(output_file, winsound.SND_FILENAME)
                    else:
                        import soundfile as sf
                        import sounddevice as sd
                        data, fs = sf.read(output_file)
                        sd.play(data, fs)
                        sd.wait()
                        
                    success = True
                    # Clean up
                    try:
                        os.remove(output_file)
                    except:
                        pass
            except Exception as e:
                print(f" [Piper Error: {e}]")
                success = False

        if not success:
            # Fallback to standard pyttsx3 if Piper failed or wasn't found
            if self.fallback_engine:
                self.fallback_engine.say(text)
                self.fallback_engine.runAndWait()
            else:
                print(f"\n[AETHER (Silent)]: {text}")
                time.sleep(1) # simulate time taken to speak if headless
                
        print(" [Done]")

if __name__ == "__main__":
    tts = PiperTTS()
    tts.speak("Aether systems online. Text to speech is functioning flawlessly.")
