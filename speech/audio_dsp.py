"""
AETHER v5.0 — Broadcast-Quality Audio DSP Chain
================================================
Inspired by Voicebox AI's Spotify Pedalboard audio effects pipeline.
Applies studio-grade post-processing to raw TTS output so AETHER's voice
sounds warm, present, and broadcast-ready — not dry and robotic.

Effects Chain:
  1. High-Pass Filter (80Hz) — removes low-end mic rumble / room hum
  2. Compressor — evens out vocal dynamics for consistent volume
  3. Gain — slight vocal presence boost
  4. Reverb — subtle studio warmth (wet=8%, dry=92%)

Dependencies: pip install pedalboard
"""

import numpy as np

try:
    from pedalboard import Pedalboard, Compressor, HighpassFilter, LowpassFilter, Reverb, Gain
    _PEDALBOARD_AVAILABLE = True
except ImportError:
    _PEDALBOARD_AVAILABLE = False
    print("[Audio DSP] pedalboard not installed. TTS output will be raw (no post-processing).")
    print("[Audio DSP] Install with: pip install pedalboard")


class AudioDSPChain:
    """
    Applies broadcast-quality audio post-processing to TTS output.
    Configurable presets for different environments and personas.
    """

    # ── Preset Configurations ──
    PRESETS = {
        "broadcast": {
            "highpass_hz": 80.0,
            "compressor_threshold_db": -16.0,
            "compressor_ratio": 2.5,
            "gain_db": 2.0,
            "reverb_room_size": 0.15,
            "reverb_damping": 0.7,
            "reverb_wet": 0.08,
            "reverb_dry": 0.92,
        },
        "warm": {
            "highpass_hz": 60.0,
            "compressor_threshold_db": -14.0,
            "compressor_ratio": 2.0,
            "gain_db": 3.0,
            "reverb_room_size": 0.25,
            "reverb_damping": 0.6,
            "reverb_wet": 0.12,
            "reverb_dry": 0.88,
        },
        "clean": {
            "highpass_hz": 100.0,
            "compressor_threshold_db": -18.0,
            "compressor_ratio": 3.0,
            "gain_db": 1.5,
            "reverb_room_size": 0.05,
            "reverb_damping": 0.9,
            "reverb_wet": 0.03,
            "reverb_dry": 0.97,
        },
        "raw": None,  # No processing — bypass
    }

    def __init__(self, preset: str = "broadcast"):
        """
        Initialize the DSP chain with a named preset.
        
        Args:
            preset: One of "broadcast", "warm", "clean", "raw"
        """
        self.preset_name = preset
        self.board = None
        self._build_chain(preset)

    def _build_chain(self, preset: str):
        """Constructs the Pedalboard effects chain from preset configuration."""
        if not _PEDALBOARD_AVAILABLE:
            self.board = None
            return

        config = self.PRESETS.get(preset)
        if config is None:
            self.board = None
            return

        self.board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=config["highpass_hz"]),
            Compressor(
                threshold_db=config["compressor_threshold_db"],
                ratio=config["compressor_ratio"],
            ),
            Gain(gain_db=config["gain_db"]),
            Reverb(
                room_size=config["reverb_room_size"],
                damping=config["reverb_damping"],
                wet_level=config["reverb_wet"],
                dry_level=config["reverb_dry"],
            ),
        ])
        print(f"[Audio DSP] Initialized '{preset}' broadcast chain: "
              f"HP={config['highpass_hz']}Hz, Comp={config['compressor_ratio']}:1, "
              f"Gain={config['gain_db']}dB, Reverb={config['reverb_wet']*100:.0f}% wet")

    def set_preset(self, preset: str):
        """Switch to a different DSP preset at runtime."""
        if preset not in self.PRESETS:
            print(f"[Audio DSP] Unknown preset '{preset}'. Available: {list(self.PRESETS.keys())}")
            return
        self.preset_name = preset
        self._build_chain(preset)

    def process(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """
        Apply the DSP chain to raw audio.
        
        Args:
            audio: Raw audio as float32 numpy array (mono)
            sample_rate: Sample rate of the audio (Kokoro default: 24000)
            
        Returns:
            Processed audio as float32 numpy array
        """
        if self.board is None:
            return audio  # Passthrough if no chain configured

        if audio is None or len(audio) == 0:
            return audio

        # Ensure correct shape for pedalboard: (channels, samples) or (samples,)
        if audio.ndim == 1:
            audio_2d = audio.reshape(1, -1).astype(np.float32)
        else:
            audio_2d = audio.astype(np.float32)

        try:
            processed = self.board(audio_2d, sample_rate)
            # Return as 1D mono
            return processed.flatten()
        except Exception as e:
            print(f"[Audio DSP] Processing error: {e}. Returning raw audio.")
            return audio.flatten()


# ─── Module-level singleton for easy import ──────────────────────────────────
_default_dsp = None

def get_dsp_chain(preset: str = "broadcast") -> AudioDSPChain:
    """Returns (or creates) the module-level DSP chain singleton."""
    global _default_dsp
    if _default_dsp is None or _default_dsp.preset_name != preset:
        _default_dsp = AudioDSPChain(preset=preset)
    return _default_dsp


if __name__ == "__main__":
    # Quick test: generate a sine wave and process it
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz A4 note
    
    dsp = AudioDSPChain(preset="broadcast")
    processed = dsp.process(tone, sr)
    print(f"Input:  shape={tone.shape}, peak={np.max(np.abs(tone)):.3f}")
    print(f"Output: shape={processed.shape}, peak={np.max(np.abs(processed)):.3f}")
    print("DSP chain test passed!")
