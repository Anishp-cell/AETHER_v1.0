"""
AETHER v5.0 — Streaming Partial STT & Fast Turn-Taking Engine
============================================================
Provides continuous streaming partial transcription during user speech:
  1. Decodes audio in rolling chunks in a lightweight background worker.
  2. Evaluates syntactic completeness on-the-fly.
  3. Signals immediate dispatch when terminal punctuation + completeness >= 0.85
     coincides with speech energy drop (<150ms latency vs standard 800ms+ silence).
"""

import time
import threading
import numpy as np
from typing import Optional, Callable
from collections import deque

try:
    from speech.dynamic_turn_taking import estimate_syntactic_completeness, get_patience_timeout
except ImportError:
    from dynamic_turn_taking import estimate_syntactic_completeness, get_patience_timeout


class StreamingSTTTracker:
    """
    Tracks streaming audio chunks and maintains an up-to-date partial
    transcription and syntactic completeness score.
    """

    def __init__(self, sample_rate: int = 16000, chunk_stride_s: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_stride_s = chunk_stride_s
        self.audio_buffer = []
        self.partial_transcript = ""
        self.completeness_score = 0.0
        self.patience_timeout = 0.9
        self.is_actively_speaking = False
        self.last_decode_time = 0.0
        self._lock = threading.Lock()

    def reset(self):
        """Reset state for a fresh utterance."""
        with self._lock:
            self.audio_buffer = []
            self.partial_transcript = ""
            self.completeness_score = 0.0
            self.patience_timeout = 0.9
            self.is_actively_speaking = False
            self.last_decode_time = time.time()

    def add_chunk(self, chunk: np.ndarray, model=None, on_partial: Optional[Callable[[str, float], None]] = None):
        """
        Appends an audio chunk and periodically triggers non-blocking partial decoding.
        """
        if chunk is None or len(chunk) == 0:
            return

        with self._lock:
            self.audio_buffer.append(chunk)
            total_samples = sum(len(c) for c in self.audio_buffer)
            current_duration_s = total_samples / self.sample_rate

        # Only decode partial every `chunk_stride_s` if model is provided
        now = time.time()
        if model is not None and (now - self.last_decode_time >= self.chunk_stride_s) and current_duration_s >= 0.6:
            self.last_decode_time = now
            
            def _async_partial():
                try:
                    with self._lock:
                        audio_flat = np.concatenate(self.audio_buffer).flatten().astype(np.float32)
                        # Normalize if int16
                        if np.max(np.abs(audio_flat)) > 1.0:
                            audio_flat = audio_flat / 32768.0

                    # Quick lightweight beam_size=1 decoding for streaming partial
                    segments, _ = model.transcribe(
                        audio_flat,
                        language="en",
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False
                    )
                    text = " ".join([s.text for s in segments]).strip()
                    
                    if text:
                        comp = estimate_syntactic_completeness(text)
                        pat = get_patience_timeout(text)
                        with self._lock:
                            self.partial_transcript = text
                            self.completeness_score = comp
                            self.patience_timeout = pat

                        if on_partial:
                            on_partial(text, comp)
                except Exception:
                    pass

            threading.Thread(target=_async_partial, daemon=True).start()

    def check_fast_dispatch_ready(self, silent_frames: int, frame_duration_ms: float = 80.0) -> bool:
        """
        Determines if the turn can be dispatched IMMEDIATELY without waiting
        for full silence timeout:
        Returns True if:
        - We have a partial transcript
        - Syntactic completeness >= 0.85 (clear complete thought)
        - Energy has dropped into silence for at least 2 frames (~160ms)
        """
        with self._lock:
            if not self.partial_transcript:
                return False
            
            # If completeness is very high (terminal punctuation or question), 2 silence frames (160ms) is enough
            if self.completeness_score >= 0.85 and silent_frames >= 2:
                return True
            
            # If completeness is moderate (>=0.6), require 5 silence frames (400ms)
            if self.completeness_score >= 0.6 and silent_frames >= 5:
                return True

            return False


# Module singleton
STREAMING_STT = StreamingSTTTracker()
