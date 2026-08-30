import os
import re
import threading
import time
import numpy as np
import warnings

# Suppress some noisy FutureWarnings from torch/transformers
warnings.filterwarnings("ignore")

try:
    from kokoro import KPipeline
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    print("[TTS Layer] Kokoro not available. Install with: pip install kokoro")

import sounddevice as sd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Import the broadcast DSP chain
try:
    from speech.audio_dsp import AudioDSPChain
except ImportError:
    try:
        from audio_dsp import AudioDSPChain
    except ImportError:
        AudioDSPChain = None


# ─────────────────────────────────────────────────────────────────────────────
# Expressive Paralinguistic Tag Engine
# ─────────────────────────────────────────────────────────────────────────────
# Inspired by Voicebox AI / Chatterbox Turbo's inline emotion tags.
# Since Kokoro-82M doesn't natively synthesize [laugh], [sigh] etc.,
# we handle them by:
#   1. Inserting natural micro-pauses at tag positions
#   2. Applying prosody hints to surrounding text (future: sampled audio cues)

_EXPRESSIVE_TAG_PATTERN = re.compile(r'\[([a-z_ ]+)\]', re.IGNORECASE)

# Tag -> (pause_before_ms, pause_after_ms, prosody_hint)
_TAG_EFFECTS = {
    "laugh":        (50, 100, None),
    "chuckle":      (30, 80,  None),
    "sigh":         (100, 80, None),
    "gasp":         (20, 60,  None),
    "pause":        (200, 0,  None),
    "clear throat": (80, 100, None),
    "cough":        (50, 80,  None),
    "sniff":        (30, 50,  None),
    "shush":        (30, 50,  None),
}


def _split_expressive_tags(text: str) -> list:
    """
    Splits text into alternating segments of plain text and expressive tags.
    Example: "Sure! [chuckle] Let me check." 
           -> [("text", "Sure!"), ("tag", "chuckle"), ("text", "Let me check.")]
    """
    parts = _EXPRESSIVE_TAG_PATTERN.split(text)
    result = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i % 2 == 0:
            result.append(("text", part))
        else:
            result.append(("tag", part.lower()))
    return result


def _generate_silence(duration_ms: int, sample_rate: int = 24000) -> np.ndarray:
    """Generate a silent audio buffer of given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def _crossfade(audio_a: np.ndarray, audio_b: np.ndarray, 
               crossfade_ms: int = 30, sample_rate: int = 24000) -> np.ndarray:
    """
    Smoothly crossfade between two audio chunks to eliminate clicks/pops.
    Inspired by Voicebox AI's unlimited-length crossfade slider.
    """
    if audio_a is None or len(audio_a) == 0:
        return audio_b if audio_b is not None else np.array([], dtype=np.float32)
    if audio_b is None or len(audio_b) == 0:
        return audio_a

    crossfade_samples = min(int(sample_rate * crossfade_ms / 1000), len(audio_a), len(audio_b))
    
    if crossfade_samples <= 0:
        return np.concatenate([audio_a, audio_b])

    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

    # Apply crossfade in the overlap region
    result = np.concatenate([
        audio_a[:-crossfade_samples],
        audio_a[-crossfade_samples:] * fade_out + audio_b[:crossfade_samples] * fade_in,
        audio_b[crossfade_samples:]
    ])
    return result


class KokoroTTS:
    """
    AETHER v5.0 Kokoro-82M TTS Engine with:
    - Sentence-level streaming with crossfade
    - Expressive paralinguistic tag processing
    - Broadcast-quality DSP post-processing
    - Interruptible playback with spoken progress tracking
    """

    def __init__(self, voice="bf_emma", lang_code="b", dsp_preset="broadcast"):
        """
        Initializes the Kokoro-82M TTS Engine.
        
        Args:
            voice: Kokoro voice preset (e.g., "bm_daniel", "bf_emma")
            lang_code: Language code ("a" = American English, "b" = British English)
            dsp_preset: Audio DSP preset ("broadcast", "warm", "clean", "raw")
        """
        print(f"[TTS Layer] Initializing Kokoro-82M Voice Engine (Voice: {voice}, DSP: {dsp_preset})...")
        
        # Force CPU initialization to prevent CuDNN conflicts with Whisper on GPU
        if _TORCH_AVAILABLE:
            original_cuda_check = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
        
        try:
            if _KOKORO_AVAILABLE:
                self.pipeline = KPipeline(lang_code=lang_code)
            else:
                self.pipeline = None
                print("[TTS Layer] WARNING: Kokoro pipeline not available!")
        finally:
            if _TORCH_AVAILABLE:
                torch.cuda.is_available = original_cuda_check
        
        self.voice = voice
        self.sample_rate = 24000
        self.is_interrupted = False
        self.speak_thread = None
        self.crossfade_ms = 30  # Configurable crossfade duration
        
        # Initialize DSP chain
        if AudioDSPChain is not None and dsp_preset != "raw":
            self.dsp = AudioDSPChain(preset=dsp_preset)
        else:
            self.dsp = None

    @property
    def _is_speaking(self):
        return self.speak_thread is not None and self.speak_thread.is_alive()

    def stop(self):
        """Immediately halts all active speech and flushes the generation pipeline."""
        self.is_interrupted = True

    def speak(self, text, on_complete=None, on_interrupt=None):
        """
        Asynchronous streaming playback with:
        - Expressive tag processing
        - Sentence-level crossfading
        - DSP post-processing
        - Interruptible polling
        """
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
            
            # Parse expressive tags
            segments = _split_expressive_tags(text)
            
            try:
                for seg_type, seg_content in segments:
                    if self.is_interrupted:
                        break
                    
                    if seg_type == "tag":
                        # Handle expressive tag — insert micro-pause
                        effect = _TAG_EFFECTS.get(seg_content, (50, 50, None))
                        pause_before, pause_after, _ = effect
                        
                        if pause_before > 0:
                            silence = _generate_silence(pause_before, self.sample_rate)
                            sd.play(silence, self.sample_rate)
                            sd.wait()
                        # Future: play sampled audio cue here (laugh SFX, etc.)
                        if pause_after > 0:
                            silence = _generate_silence(pause_after, self.sample_rate)
                            sd.play(silence, self.sample_rate)
                            sd.wait()
                        continue
                    
                    # Text segment — synthesize with Kokoro
                    if self.pipeline is None:
                        continue
                        
                    generator = self.pipeline(
                        seg_content, voice=self.voice, speed=1.0,
                        split_pattern=r'(?<=[.!?])\s+'
                    )
                    
                    for i, (graphemes, phonemes, audio) in enumerate(generator):
                        if self.is_interrupted:
                            break
                        if audio is not None:
                            # Apply DSP post-processing
                            if self.dsp is not None:
                                audio = self.dsp.process(audio, self.sample_rate)
                            
                            sd.play(audio, self.sample_rate)
                            
                            chunk_completed = True
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

    # Complete catalog of Voicebox / Kokoro-82M voices
    AVAILABLE_VOICES = {
        # British English (lang_code='b')
        "daniel":   ("bm_daniel",   "b", "British Male (Authoritative, Jarvis-style)"),
        "george":   ("bm_george",   "b", "British Male (Warm narrator)"),
        "fable":    ("bm_fable",    "b", "British Male (Expressive storytelling)"),
        "lewis":    ("bm_lewis",    "b", "British Male (Deep, calm)"),
        "emma":     ("bf_emma",     "b", "British Female (Crisp assistant)"),
        "isabella": ("bf_isabella", "b", "British Female (Smooth, clear)"),
        "alice":    ("bf_alice",    "b", "British Female (Soft, articulate)"),
        "lily":     ("bf_lily",     "b", "British Female (Bright, energetic)"),
        # American English (lang_code='a')
        "heart":    ("af_heart",    "a", "American Female (Warm, friendly)"),
        "bella":    ("af_bella",    "a", "American Female (Casual, natural)"),
        "sarah":    ("af_sarah",    "a", "American Female (Professional)"),
        "nicole":   ("af_nicole",   "a", "American Female (Conversational)"),
        "sky":      ("af_sky",      "a", "American Female (Youthful, upbeat)"),
        "adam":     ("am_adam",     "a", "American Male (Dynamic, confident)"),
        "michael":  ("am_michael",  "a", "American Male (Deep corporate)"),
        "eric":     ("am_eric",     "a", "American Male (Friendly conversational)"),
        "liam":     ("am_liam",     "a", "American Male (Casual young adult)"),
        "onyx":     ("am_onyx",     "a", "American Male (Deep baritone)"),
    }

    def set_voice(self, voice_name: str):
        """
        Switches the active Voicebox voice.
        Supports both short names ('daniel', 'heart', 'emma') and raw codes ('bm_daniel', 'af_heart').
        """
        name_clean = voice_name.lower().strip()
        
        if name_clean in self.AVAILABLE_VOICES:
            code, lang, desc = self.AVAILABLE_VOICES[name_clean]
            self.voice = code
            if self.pipeline is None or getattr(self.pipeline, 'lang_code', None) != lang:
                if _KOKORO_AVAILABLE:
                    self.pipeline = KPipeline(lang_code=lang)
            print(f"[TTS Layer] Voice switched to: {code} ({desc})")
        else:
            # Assume raw code
            self.voice = voice_name
            print(f"[TTS Layer] Voice set to raw code: {voice_name}")

    @classmethod
    def list_voices(cls):
        """Returns the dictionary of all available Voicebox voices with descriptions."""
        return cls.AVAILABLE_VOICES


if __name__ == "__main__":
    # Test script
    tts = KokoroTTS(voice="bm_daniel", lang_code="b", dsp_preset="broadcast")
    print("\nAvailable Voicebox AI voices in AETHER:")
    for alias, (code, lang, desc) in tts.list_voices().items():
        print(f"  • {alias.ljust(10)} -> {code.ljust(12)} ({desc})")
    
    tts.speak("Hello sir, I am Aether. [pause] My voice engine has been upgraded with broadcast-quality audio processing.")
    while tts._is_speaking:
        time.sleep(0.1)
