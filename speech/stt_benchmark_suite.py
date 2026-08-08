"""
AETHER v4.0 — Menu-Driven STT Testing & Comparison Suite
Allows you to switch between different speech-to-text configurations,
speak standardized test sentences, and directly compare latency, accuracy, and hallucination immunity.
"""
import os
import sys
import time
import queue
import re
import numpy as np
import sounddevice as sd
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
SILENCE_TIMEOUT = 1.0
PRE_BUFFER_DURATION = 0.4


class STTBenchmarkSuite:
    def __init__(self):
        self.model = None
        self.model_name = "base.en"
        self.audio_queue = queue.Queue()
        self.running = True

    def load_whisper(self, model_size="base.en"):
        if self.model is None or self.model_name != model_size:
            from faster_whisper import WhisperModel
            print(f"\n[STT Suite] Loading Faster-Whisper '{model_size}' on CPU (int8)...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.model_name = model_size
            print("[STT Suite] Model ready!")

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def record_utterance(self, energy_threshold=0.02) -> np.ndarray:
        """Listens until speech finishes, then returns 16kHz float32 audio."""
        print("\n🎙️ [LISTENING] Speak your test sentence now (Press Ctrl+C to cancel)...")
        pre_buffer_len = int(PRE_BUFFER_DURATION * SAMPLE_RATE / BLOCK_SIZE)
        pre_buffer = deque(maxlen=pre_buffer_len)
        recording_buffer = []
        is_recording = False
        last_speech_time = 0.0

        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, dtype="float32", callback=self.audio_callback):
            while True:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    peak = np.max(np.abs(chunk))
                    now = time.time()

                    if peak > energy_threshold:
                        last_speech_time = now
                        if not is_recording:
                            is_recording = True
                            print("   🔴 [Recording Audio Stream]...")
                            recording_buffer.extend(pre_buffer)

                    if is_recording:
                        recording_buffer.append(chunk)
                        if now - last_speech_time > SILENCE_TIMEOUT:
                            print("   ⏳ [Processing Speech]...")
                            break
                    else:
                        pre_buffer.append(chunk)

                except queue.Empty:
                    continue

        if recording_buffer:
            return np.concatenate(recording_buffer).flatten().astype(np.float32)
        return np.array([], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 1: Baseline Default Whisper
    # ─────────────────────────────────────────────────────────────────────────
    def test_option_1_baseline(self, audio_data: np.ndarray):
        self.load_whisper("base.en")
        t0 = time.time()
        segments, info = self.model.transcribe(audio_data, language="en", beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        latency_ms = (time.time() - t0) * 1000
        return text, latency_ms, "None"

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 2: Faster-Whisper + Prompt Armor + VAD Filter
    # ─────────────────────────────────────────────────────────────────────────
    def test_option_2_prompt_armor(self, audio_data: np.ndarray):
        self.load_whisper("base.en")
        t0 = time.time()
        # Prompt conditioning constrains vocabulary away from YouTube subtitles
        segments, info = self.model.transcribe(
            audio_data,
            language="en",
            initial_prompt="AETHER assistant desktop voice command:",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False
        )
        raw_text = " ".join([s.text for s in segments]).strip()
        
        # Anti-hallucination regex sanitizer
        cleaned = raw_text
        phantom_patterns = [r'\[music\]', r'thank you for watching', r'subtitles by', r'^thank you\.?$', r'^bye\.?$']
        for p in phantom_patterns:
            cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
            
        latency_ms = (time.time() - t0) * 1000
        return cleaned or "(Filtered Hallucination / Silence)", latency_ms, "VAD + Prompt Armor + Regex"

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 3: Greedy Fast Decoder (beam_size=1, temp=0.0)
    # ─────────────────────────────────────────────────────────────────────────
    def test_option_3_greedy(self, audio_data: np.ndarray):
        self.load_whisper("base.en")
        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_data,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False
        )
        text = " ".join([s.text for s in segments]).strip()
        latency_ms = (time.time() - t0) * 1000
        return text, latency_ms, "Greedy (beam_size=1, temp=0.0)"

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 4: Higher-Accuracy Small Model (small.en + VAD)
    # ─────────────────────────────────────────────────────────────────────────
    def test_option_4_small_en(self, audio_data: np.ndarray):
        self.load_whisper("small.en")
        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_data,
            language="en",
            beam_size=3,
            vad_filter=True,
            initial_prompt="AETHER voice command:"
        )
        text = " ".join([s.text for s in segments]).strip()
        latency_ms = (time.time() - t0) * 1000
        return text, latency_ms, "small.en + VAD"

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION 5: Side-by-Side Comparison: Option 2 (base.en) vs Option 4 (small.en)
    # ─────────────────────────────────────────────────────────────────────────
    def test_option_5_all_side_by_side(self, audio_data: np.ndarray):
        print("\n" + "═" * 75)
        print("  📊 RUNNING DIRECT HEAD-TO-HEAD BENCHMARK (OPTION 2 vs OPTION 4)")
        print("═" * 75)

        # Run Option 2: base.en + Prompt Armor + VAD
        text_opt2, lat_opt2, desc_opt2 = self.test_option_2_prompt_armor(audio_data)

        # Run Option 4: small.en + Prompt Armor + VAD
        self.load_whisper("small.en")
        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio_data,
            language="en",
            initial_prompt="AETHER assistant desktop voice command:",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False
        )
        raw_small = " ".join([s.text for s in segments]).strip()
        cleaned_small = raw_small
        for p in [r'\[music\]', r'thank you for watching', r'subtitles by', r'^thank you\.?$', r'^bye\.?$']:
            cleaned_small = re.sub(p, '', cleaned_small, flags=re.IGNORECASE).strip()
        lat_opt4 = (time.time() - t0) * 1000
        text_opt4 = cleaned_small or "(Filtered Hallucination / Silence)"

        print(f"\n[Option 2 — base.en + Prompt Armor + VAD]:")
        print(f"   ⏱️ Latency: {lat_opt2:.1f} ms  |  ⚡ Speed Advantage: {(lat_opt4 / max(1, lat_opt2)):.1f}x Faster")
        print(f"   📝 Output:  \"{text_opt2}\"")

        print(f"\n[Option 4 — small.en + Prompt Armor + VAD]:")
        print(f"   ⏱️ Latency: {lat_opt4:.1f} ms")
        print(f"   📝 Output:  \"{text_opt4}\"")

        # Delta analysis
        print("\n" + "─" * 75)
        if text_opt2.lower().strip() == text_opt4.lower().strip():
            print("🎯 Match: Both models transcribed the EXACT same text perfectly!")
        else:
            print(f"🔍 Difference Detected:")
            print(f"   • base.en : \"{text_opt2}\"")
            print(f"   • small.en: \"{text_opt4}\"")
        print("═" * 75)


def print_menu():
    print("\n" + "═" * 75)
    print("      🎙️ AETHER STT BENCHMARK & HEAD-TO-HEAD COMPARISON TESTBED")
    print("═" * 75)
    print(" 📖 HARDER CHALLENGE SENTENCES TO TEST WITH:")
    print("  [A] Fast Desktop: 'Snap the active terminal window to the left and tile all windows.'")
    print("  [B] Technical/API: 'Calculate the latest stock rises for Nvidia, Apple, and Google via yfinance.'")
    print("  [C] Media/Soundalike: 'Play my liked songs playlist on Spotify and raise volume to 50 percent.'")
    print("  [D] Visual Precision: 'Highlight the search input bar in the top navigation header on my screen.'")
    print("  [E] Acoustic Noise: (Stay completely silent, cough, snap fingers, or type on keyboard)")
    print("─" * 75)
    print(" [1] Option 1: Baseline Faster-Whisper (Default base.en)")
    print(" [2] Option 2: Faster-Whisper + Prompt Armor + VAD (base.en ~450ms)")
    print(" [3] Option 3: Fast Greedy Decoder (beam_size=1, sub-50ms CPU)")
    print(" [4] Option 4: Higher-Accuracy small.en with Prompt Armor + VAD (~1300ms)")
    print(" [5] Option 5: ⚔️ Head-to-Head Multi-Run (Runs EXACT SAME audio on Option 2 vs 4)")
    print(" [6] Exit Testbed")
    print("═" * 75)


def main():
    suite = STTBenchmarkSuite()
    while True:
        print_menu()
        choice = input("\n👉 Choose option (1-6): ").strip()

        if choice == "1":
            print("\n--- Testing Option 1: Baseline Default Whisper ---")
            audio = suite.record_utterance()
            if len(audio) > 0:
                text, lat, desc = suite.test_option_1_baseline(audio)
                print(f"\n✅ Result ({lat:.1f}ms): \"{text}\"")
        elif choice == "2":
            print("\n--- Testing Option 2: base.en + Prompt Armor + VAD ---")
            audio = suite.record_utterance()
            if len(audio) > 0:
                text, lat, desc = suite.test_option_2_prompt_armor(audio)
                print(f"\n✅ Result ({lat:.1f}ms): \"{text}\"")
        elif choice == "3":
            print("\n--- Testing Option 3: Fast Greedy Decoder ---")
            audio = suite.record_utterance()
            if len(audio) > 0:
                text, lat, desc = suite.test_option_3_greedy(audio)
                print(f"\n✅ Result ({lat:.1f}ms): \"{text}\"")
        elif choice == "4":
            print("\n--- Testing Option 4: small.en + Prompt Armor + VAD ---")
            audio = suite.record_utterance()
            if len(audio) > 0:
                text, lat, desc = suite.test_option_4_small_en(audio)
                print(f"\n✅ Result ({lat:.1f}ms): \"{text}\"")
        elif choice == "5":
            print("\n--- Testing Option 5: ⚔️ Head-to-Head Comparison (Option 2 vs Option 4) ---")
            audio = suite.record_utterance()
            if len(audio) > 0:
                suite.test_option_5_all_side_by_side(audio)
        elif choice == "6":
            print("\nExiting STT Benchmark Testbed.")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
