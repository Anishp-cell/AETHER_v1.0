"""
AETHER v5.0 — Voicebox AI Voice Showcase & Interactive Tester
=============================================================
Quickly test and sample all 18 Voicebox / Kokoro-82M voices right inside AETHER!
Allows testing:
  1. Voice Switching (Daniel, Heart, Emma, Onyx, George, Bella, etc.)
  2. DSP Studio Presets (Broadcast, Warm, Clean, Raw)
  3. Expressive Emotion Tags ([laugh], [sigh], [pause], [chuckle])
"""

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "speech"))

from TTS_kokoro import KokoroTTS


def test_voice_sample(voice_alias: str, dsp_preset: str = "broadcast"):
    """Samples a single voice with a natural test phrase."""
    print(f"\n" + "=" * 60)
    print(f"  Testing Voice: '{voice_alias}' (DSP: {dsp_preset})")
    print("=" * 60)

    tts = KokoroTTS(voice="bm_daniel", dsp_preset=dsp_preset)
    tts.set_voice(voice_alias)

    sample_phrase = (
        f"Hello sir, this is the {voice_alias} voice. "
        f"[pause] All Aether cognitive and acoustic systems are operational. "
        f"[chuckle] How can I assist you today?"
    )

    print(f"Phrase: \"{sample_phrase}\"")
    tts.speak(sample_phrase)

    while tts._is_speaking:
        time.sleep(0.1)


def interactive_voice_menu():
    """Interactive CLI menu to audition voices."""
    print("=" * 70)
    print("  VOICEBOX AI / KOKORO-82M VOICE AUDITION STUDIO")
    print("=" * 70)
    
    voices = KokoroTTS.list_voices()
    aliases = list(voices.keys())

    for idx, (alias, (code, lang, desc)) in enumerate(voices.items(), 1):
        print(f"  [{idx:02d}] {alias.ljust(10)} ({code.ljust(12)}) - [{lang.upper()}] {desc}")

    print("\nPresets available: broadcast, warm, clean, raw")
    print("Commands:")
    print("  • Type a number (1-18) or voice name to audition that voice.")
    print("  • Type 'all' to audition top 4 recommended voices sequentially.")
    print("  • Type 'q' to exit.")

    while True:
        try:
            choice = input("\nSelect voice (or 'q' to quit) > ").strip().lower()
            if not choice or choice in ['q', 'exit', 'quit']:
                print("Exiting Voice Audition Studio.")
                break
            
            if choice == 'all':
                for v in ["daniel", "heart", "emma", "onyx"]:
                    test_voice_sample(v)
                    time.sleep(0.5)
                continue

            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(aliases):
                    selected = aliases[idx - 1]
                    test_voice_sample(selected)
                else:
                    print(f"Invalid index. Choose 1 to {len(aliases)}.")
            elif choice in voices:
                test_voice_sample(choice)
            else:
                print(f"Unknown voice '{choice}'. Available: {', '.join(aliases)}")
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_voice_sample(sys.argv[1])
    else:
        interactive_voice_menu()
