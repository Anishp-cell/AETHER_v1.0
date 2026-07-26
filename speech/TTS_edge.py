import os
import threading
import time
import uuid
import pygame
import asyncio
import edge_tts

class TTSEngine:
    def __init__(self, voice="en-GB-RyanNeural"):
        print(f"[TTS Layer] Initializing Edge-TTS (Cloud Engine, Voice: {voice})...")
        self.voice = voice
        pygame.mixer.init()
        self._is_speaking = False
        self._lock = threading.Lock()
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self._temp_dir = os.path.join(base_dir, "temp_audio")
        os.makedirs(self._temp_dir, exist_ok=True)

    def speak(self, text: str, on_complete=None, on_interrupt=None):
        if not text.strip():
            if on_complete: on_complete()
            return

        def _speak_thread():
            with self._lock:
                self._is_speaking = True
                # Unique file per utterance to prevent Windows file-locking conflicts
                output_file = os.path.join(self._temp_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3")
                
                try:
                    async def generate():
                        communicate = edge_tts.Communicate(text, self.voice)
                        await communicate.save(output_file)
                    asyncio.run(generate())
                except Exception as e:
                    print(f"[TTS Layer] Edge-TTS API failed: {e}")
                    self._is_speaking = False
                    if on_complete: on_complete()
                    return

                print("[Speaking]...", end="", flush=True)

                try:
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy() and self._is_speaking:
                        time.sleep(0.05)
                        
                    if not self._is_speaking:
                        pygame.mixer.music.stop()
                        print("\n [Interrupted]")
                        if on_interrupt:
                            on_interrupt(text[:len(text)//2])
                    else:
                        print(" [Done]")
                        if on_complete: on_complete()
                        
                except Exception as e:
                    print(f"[TTS Error] Pygame Playback failed: {e}")
                    if on_complete: on_complete()
                finally:
                    self._is_speaking = False
                    # CRITICAL: Release the file handle so future writes don't get Permission Denied
                    try:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
                    try:
                        os.remove(output_file)
                    except OSError:
                        pass

        threading.Thread(target=_speak_thread, daemon=True).start()

    def halt(self):
        self._is_speaking = False

    def stop(self):
        """Alias for halt - called by server.py on voice interruption."""
        self.halt()
