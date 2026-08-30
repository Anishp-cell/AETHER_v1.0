"""
AETHER v5.0 — Contextual Interruption Recovery Manager
======================================================
Implements human-like conversational resumption:
When AETHER is interrupted mid-speech:
  1. Captures the unsaid remainder of the utterance and extracts key topic terms.
  2. Evaluates the user's interruption (was it a hard abort like "stop/cancel",
     or a quick tangent/clarification like "wait, what version?").
  3. If it was a tangent, enables AETHER to answer the tangent and then naturally
     offer to resume: "Would you like me to continue where I left off regarding [topic]?"
  4. If user agrees ("yes", "sure", "continue", "go on"), smoothly resumes the unsaid portion.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class InterruptedContext:
    """Stores the state of an interrupted AETHER response."""
    original_full_text: str
    spoken_so_far: str
    remaining_text: str
    topic_summary: str
    timestamp: float = field(default_factory=time.time)
    resumption_offered: bool = False
    is_active: bool = True

    @property
    def remaining_word_count(self) -> int:
        return len(self.remaining_text.split())

    @property
    def is_worth_resuming(self) -> bool:
        """Only offer resumption if there was substantial unsaid content (>12 words)."""
        return self.is_active and self.remaining_word_count >= 12 and (time.time() - self.timestamp < 120.0)


def extract_topic_summary(text: str) -> str:
    """
    Extracts a concise 2-4 word topic label from the text to use in
    resumption prompts (e.g. 'the system diagnostics', 'the Python script').
    Uses flexible regex noun-phrase heuristics.
    """
    if not text:
        return "that"

    clean = text.strip()
    
    # Try finding subject after common introductory phrases
    patterns = [
        r'(?:regarding|about|explaining|details on|information about)\s+([a-zA-Z0-9_\-\s]{3,25})(?:[,.\n]|$)',
        r'(?:the|your)\s+([a-zA-Z0-9_\-\s]{3,20})\s+(?:is|are|was|were|has|have)',
        r'^(?:Here (?:is|are)|This is)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]{3,20})',
    ]
    for p in patterns:
        m = re.search(p, clean, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            # Clean trailing words
            topic = re.sub(r'\s+(?:and|or|that|which|with|for)$', '', topic, flags=re.IGNORECASE)
            if len(topic.split()) <= 4:
                return topic.lower()

    # Exclude polite pleasantries and assistant tropes from topic extraction
    _STOP_WORDS = {
        "there", "which", "about", "their", "would", "could", "should", "these", "those",
        "delighted", "assist", "assisting", "pleasure", "happy", "welcome", "certainly",
        "course", "hello", "here", "great", "sure", "simple", "effective", "first", "case"
    }

    # Fallback: take first significant words that aren't assistant pleasantries
    words = [w.strip(".,!?\"'") for w in clean.split() if len(w) > 3 and w.lower() not in _STOP_WORDS]
    if len(words) >= 2:
        return f"{words[0]} {words[1]}".lower()
    elif len(words) == 1:
        return words[0].lower()
    
    return "our previous discussion"


class InterruptionRecoveryManager:
    """
    Manages conversational memory of interrupted responses and handles
    contextual resumption.
    """

    def __init__(self):
        self.last_interrupted: Optional[InterruptedContext] = None

    def record_interruption(self, original_full_text: str, spoken_so_far: str) -> Optional[InterruptedContext]:
        """
        Called when TTS playback is interrupted by the user.
        Calculates what was left unsaid.
        """
        if not original_full_text or not original_full_text.strip():
            self.last_interrupted = None
            return None

        # Clean strings
        full_clean = original_full_text.strip()
        spoken_clean = spoken_so_far.strip() if spoken_so_far else ""

        # Find where speech stopped
        remaining = ""
        if spoken_clean and spoken_clean in full_clean:
            idx = full_clean.find(spoken_clean) + len(spoken_clean)
            remaining = full_clean[idx:].strip()
        else:
            # Approximate remaining based on word length
            spoken_words = len(spoken_clean.split())
            all_words = full_clean.split()
            if spoken_words < len(all_words):
                remaining = " ".join(all_words[spoken_words:]).strip()
            else:
                remaining = ""

        # Clean leading punctuation from remaining
        remaining = re.sub(r'^[,\s.:;!?-]+', '', remaining).strip()

        topic = extract_topic_summary(full_clean)

        self.last_interrupted = InterruptedContext(
            original_full_text=full_clean,
            spoken_so_far=spoken_clean,
            remaining_text=remaining,
            topic_summary=topic,
            timestamp=time.time(),
            resumption_offered=False,
            is_active=True
        )

        print(f"  [Interruption Recovery] Saved context: topic='{topic}', remaining={self.last_interrupted.remaining_word_count} words")
        return self.last_interrupted

    def check_resumption_agreement(self, user_text: str) -> Optional[str]:
        """
        Checks if the user's latest input is an affirmative agreement to resume
        a previously offered interrupted context (e.g. 'yes', 'please', 'continue', 'go on').
        
        Returns:
            The remaining text to speak, or None if not an agreement.
        """
        if not self.last_interrupted or not self.last_interrupted.is_active:
            return None

        # Expire after 90 seconds
        if time.time() - self.last_interrupted.timestamp > 90.0:
            self.clear()
            return None

        clean_user = user_text.strip().lower()

        # Check for affirmative / resumption signals (flexible combinations)
        affirmative_pattern = (
            r'^(?:'
            r'(?:yes|yeah*|yep|yup|sure|ok(?:ay)?|please|definitely|absolutely)'
            r'(?:\s+(?:please|continue|go\s+(?:ahead|on)|keep\s+going|resume|do\s+that))*'
            r'|(?:continue|go\s+(?:ahead|on)|keep\s+going|finish|resume|proceed|what\s+else|tell\s+me)'
            r'(?:\s+please)?'
            r')\s*[.!]?$'
        )

        if re.match(affirmative_pattern, clean_user):
            remaining = self.last_interrupted.remaining_text
            self.clear()
            print(f"  [Interruption Recovery] User agreed to resume! Delivering {len(remaining.split())} words.")
            return remaining

        # If user clearly says "no / stop / forget it / never mind", discard the interrupted context
        dismiss_pattern = r'^(?:no(?: thanks)?|nah|stop|cancel|never ?mind|forget it|ignore that|drop it)\s*[.!]?$'
        if re.match(dismiss_pattern, clean_user):
            print(f"  [Interruption Recovery] User dismissed previous topic.")
            self.clear()
            return None

        return None

    def format_reply_with_resumption_cue(self, current_reply: str, user_text: str) -> str:
        """
        Appends a gentle, natural resumption cue to AETHER's current reply if:
        1. An interrupted context with substantial content is pending.
        2. The user's input was a brief question or tangent, not a cancellation.
        """
        if not self.last_interrupted or not self.last_interrupted.is_worth_resuming:
            return current_reply

        # Don't offer if already offered
        if self.last_interrupted.resumption_offered:
            return current_reply

        # Check if user input was a command to do something completely new
        user_lower = user_text.lower()
        if any(w in user_lower for w in ["open ", "close ", "search ", "play ", "send ", "create ", "delete "]):
            # User moved on to a new action, don't interrupt them
            self.clear()
            return current_reply

        self.last_interrupted.resumption_offered = True
        topic = self.last_interrupted.topic_summary

        cue = f" Would you like me to continue where I left off regarding {topic}?"
        return current_reply.rstrip() + cue

    def clear(self):
        """Clears any pending interrupted context."""
        if self.last_interrupted:
            self.last_interrupted.is_active = False
            self.last_interrupted = None


# Singleton instance
INTERRUPTION_RECOVERY = InterruptionRecoveryManager()
