"""
AETHER v5.0 — Tiered Response Controller
=========================================
Classifies AETHER's outgoing responses into three conversational tiers
and adjusts the delivery style (speed, verbosity, pauses) accordingly.

Tier 1: Quick Verbal Feedback (<3 sec)
  - "Done.", "Got it.", "Playing now."
  - Direct confirmations, boolean answers, status updates
  - Delivered at 1.1x speed, no preamble

Tier 2: Standard Conversational Response (3-15 sec)
  - Normal assistant answers, explanations, task confirmations
  - Natural speaking pace with sentence breaks

Tier 3: Extended Narrative / Complex Output (>15 sec)
  - Long explanations, code walkthroughs, multi-step instructions
  - Spoken with natural pauses, and a preemptive "This might take a moment" cue
"""

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ResponseTier(IntEnum):
    QUICK = 1      # <30 words, direct confirmations
    STANDARD = 2   # 30-100 words, normal answers
    EXTENDED = 3   # 100+ words, long narratives


@dataclass
class TieredResponse:
    """A response annotated with delivery metadata."""
    tier: ResponseTier
    text: str              # Full response text
    spoken_text: str       # Text optimized for TTS
    speed: float           # TTS speed multiplier (1.0 = normal)
    preamble: Optional[str]  # Optional "heads up" before the main content


def classify_and_format(reply: str, intent: str = "") -> TieredResponse:
    """
    Classifies an LLM reply into a response tier and formats it for
    natural spoken delivery.
    
    Args:
        reply: Raw LLM response text
        intent: The detected user intent (e.g., "desktop command", "casual chat")
        
    Returns:
        TieredResponse with tier classification and delivery parameters
    """
    if not reply or not reply.strip():
        return TieredResponse(
            tier=ResponseTier.QUICK, text="", spoken_text="",
            speed=1.0, preamble=None
        )

    # Count words (rough proxy for speaking duration)
    words = reply.split()
    word_count = len(words)
    
    # ── Tier 1: Quick Verbal Feedback ──
    # Short confirmations, yes/no, status updates
    if word_count <= 30:
        return TieredResponse(
            tier=ResponseTier.QUICK,
            text=reply,
            spoken_text=reply.strip(),
            speed=1.05,  # Slightly faster for snappy delivery
            preamble=None
        )
    
    # ── Tier 3: Extended Narrative ──
    # Long answers that would take >15 seconds to speak
    if word_count > 100:
        # For very long responses, consider summarizing the spoken version
        spoken = reply.strip()
        
        # If it has numbered lists or bullet points, it's likely a walkthrough
        has_structure = bool(re.search(r'(?:^\d+\.|^[-•])\s', reply, re.MULTILINE))
        
        preamble = None
        if word_count > 200 and has_structure:
            preamble = "Here's a detailed breakdown."
        elif word_count > 150:
            preamble = "Let me explain."
        
        return TieredResponse(
            tier=ResponseTier.EXTENDED,
            text=reply,
            spoken_text=spoken,
            speed=0.95,  # Slightly slower for clarity on complex content
            preamble=preamble
        )
    
    # ── Tier 2: Standard Conversational Response ──
    return TieredResponse(
        tier=ResponseTier.STANDARD,
        text=reply,
        spoken_text=reply.strip(),
        speed=1.0,
        preamble=None
    )


def format_for_speech(reply: str, intent: str = "") -> str:
    """
    Convenience function: takes raw LLM output and returns speech-ready text.
    Prepends preamble if needed, strips non-spoken syntax.
    
    This is the primary entry point for server.py integration.
    """
    tiered = classify_and_format(reply, intent)
    
    result = tiered.spoken_text
    if tiered.preamble:
        result = f"{tiered.preamble} {result}"
    
    return result
