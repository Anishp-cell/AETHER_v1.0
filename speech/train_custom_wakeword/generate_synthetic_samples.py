"""
AETHER v4.0 — Synthetic Wake-Word Audio Generator
Generates multi-speaker, multi-pitch, and multi-speed synthetic audio samples
for training custom openWakeWord models (e.g., 'Hey AETHER', 'AETHER').
"""
import os
import sys
import asyncio
import wave
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PHRASES = [
    "Hey AETHER",
    "AETHER",
    "Hey Ether",
    "Ay-ther",
    "AETHER online",
    "Hey AETHER wake up",
    "AETHER status",
]

VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
]

RATES = ["-15%", "-5%", "+0%", "+10%", "+20%"]
PITCHES = ["-10Hz", "+0Hz", "+10Hz", "+20Hz"]


async def generate_single_clip(phrase: str, voice: str, rate: str, pitch: str, filename: str):
    """Synthesizes a single audio clip using edge_tts with specified cadence."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(phrase, voice, rate=rate, pitch=pitch)
        await communicate.save(filename)
        return True
    except Exception as e:
        return False


async def generate_dataset(max_samples: int = 150):
    """Generates synthetic dataset of target wake phrases."""
    print(f"🎙️ [WakeWord Generator] Generating {max_samples} synthetic training samples in '{OUTPUT_DIR}'...")
    count = 0
    tasks = []

    for phrase in TARGET_PHRASES:
        for voice in VOICES:
            for rate in RATES:
                for pitch in PITCHES:
                    if count >= max_samples:
                        break
                    out_path = os.path.join(OUTPUT_DIR, f"sample_{count:04d}_{voice[:5]}_{phrase.replace(' ', '_')}.mp3")
                    tasks.append(generate_single_clip(phrase, voice, rate, pitch, out_path))
                    count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break
        if count >= max_samples:
            break

    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if r)
    print(f"✅ [WakeWord Generator] Generated {success_count}/{count} synthetic clips for 'Hey AETHER'!")
    return success_count


if __name__ == "__main__":
    asyncio.run(generate_dataset(150))
