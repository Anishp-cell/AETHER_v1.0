import pyttsx3
import threading

class TextToSpeech:
    def __init__(self, rate=150, volume=1.0):
        """
        Initialize the TTS engine.
        :param rate: Speed of speech.
        :param volume: Volume level (0.0 to 1.0).
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Optionally set voice (0 is usually default male, 1 is female in Windows)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)

    def speak(self, text):
        """
        Speak the given text blockingly.
        """
        print(f"[AETHER]: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def speak_async(self, text):
        """
        Speak the given text non-blockingly using a background thread.
        """
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Hello, I am AETHER. Initialization sequence complete.")
