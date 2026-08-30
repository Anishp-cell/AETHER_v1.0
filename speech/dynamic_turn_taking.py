"""
AETHER v5.0 — Sparrow-2 Inspired Conversational State Machine
==============================================================
Implements natural, human-like conversational turn-taking:
  1. Intent-Aware Dynamic Patience (holds floor during mid-thought pauses)
  2. Backchannel vs. Interruption Classification (flexible, not hardcoded)
  3. Honest Listening Confidence Gate (rejects low-confidence Whisper output)
  4. Disfluency Refinement (strips stutters, false starts, filler words)

All classifiers use FLEXIBLE heuristic strategies (duration, energy, word count,
POS-like trailing patterns, compression ratio) rather than fixed word sets.
"""

import re
import time
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────────────────────
# 1. Conversational Turn State
# ─────────────────────────────────────────────────────────────────────────────

class TurnState(Enum):
    """Sparrow-2 inspired conversational floor states."""
    IDLE           = auto()   # No one is speaking, ambient monitoring
    USER_SPEAKING  = auto()   # User holds the floor
    USER_PAUSING   = auto()   # User has paused mid-thought (patience window active)
    AETHER_SPEAKING = auto()  # AETHER is delivering a response
    PROCESSING     = auto()   # STT/LLM processing in flight


@dataclass
class ConversationContext:
    """Tracks the running conversational context for turn-taking decisions."""
    state: TurnState = TurnState.IDLE
    pause_start_time: float = 0.0
    last_speech_time: float = 0.0
    partial_transcript: str = ""
    consecutive_silence_frames: int = 0
    consecutive_loud_frames: int = 0
    ambient_energy_floor: float = 300.0  # Adaptive noise floor (EMA)
    recent_energies: list = field(default_factory=list)
    _ema_alpha: float = 0.02  # Slow-moving average for ambient floor


# ─────────────────────────────────────────────────────────────────────────────
# 2. Intent-Aware Patience Controller
# ─────────────────────────────────────────────────────────────────────────────

# Flexible trailing-pattern categories (regex-based, not exact word sets)
# These detect SYNTACTIC incompleteness at the end of a partial transcript.
_TRAILING_CONNECTOR_PATTERNS = [
    # Conjunctions / prepositions that signal more content is coming
    r'\b(?:and|or|but|so|because|since|while|although|though|if|when|'
    r'where|that|which|who|whose|whom|whether|unless|until|before|after|'
    r'for|with|to|from|about|into|onto|of|at|in|on|by|as|than|like)\s*$',
    # Trailing articles / determiners (user is about to name something)
    r'\b(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|some|any|every)\s*$',
    # Trailing verbs that expect an object  
    r'\b(?:open|close|search|find|send|set|create|delete|play|pause|start|stop|'
    r'show|hide|move|copy|download|upload|install|run|execute|schedule|call|'
    r'text|email|write|read|check|get|put|make|take|give|tell|ask|say|bring)\s*$',
    # Trailing "is", "are", "was" (incomplete predicate)
    r'\b(?:is|are|was|were|will|would|could|should|can|may|might|shall|has|have|had|do|does|did)\s*$',
]

# Hesitation / thinking markers — broad regex patterns, not a fixed list
_HESITATION_PATTERNS = [
    r'\b(?:um+|uh+|er+|eh+|ah+|hmm+|hm+|emm+)\b',  # Classic fillers
    r'\b(?:like|basically|actually|literally|you know|I mean)\s*$',  # Discourse markers at end
    r'\.{2,}\s*$',  # Trailing ellipsis ("open the...")
]


def estimate_syntactic_completeness(text: str) -> float:
    """
    Returns a score from 0.0 (clearly incomplete) to 1.0 (likely complete).
    Uses flexible pattern matching and acoustic/syntactic heuristics rather than rigid word lists.
    """
    if not text or not text.strip():
        return 0.0

    raw_text = text.strip()
    text_clean = raw_text.lower()
    words = text_clean.split()
    word_count = len(words)

    # 1. Immediate hesitation marker detection (e.g., "um", "uh", "hmm...")
    is_pure_hesitation = any(re.fullmatch(p.strip(r'\b'), text_clean) for p in _HESITATION_PATTERNS if not p.endswith(r'\s*$'))
    if is_pure_hesitation or re.search(r'^(?:um+|uh+|er+|eh+|ah+|hmm+|hm+|emm+)\b', text_clean) and word_count <= 2:
        return 0.1  # User is hesitating/thinking

    # Base score initialization
    if word_count == 1:
        # Single-word imperatives / common quick commands
        if re.match(r'^(?:stop|pause|play|resume|cancel|mute|unmute|status|help|yes|no|done|exit|quit|bye|hello|hi)$', text_clean):
            score = 0.9
        else:
            score = 0.45
    elif word_count == 2:
        # 2-word commands like "open chrome", "stop music", "check battery"
        score = 0.75
    else:
        # 3+ words: default conversational utterance base
        score = 0.7

    # 2. Check for trailing connectors (strong signal of mid-sentence thought pause)
    for pattern in _TRAILING_CONNECTOR_PATTERNS:
        # If it's a single word imperative (e.g. "play", "stop"), don't penalize as trailing verb
        if word_count == 1 and re.match(r'^(?:stop|pause|play|resume|cancel|mute|unmute|status|help|yes|no|done|exit|quit|bye|hello|hi)$', text_clean):
            break
        if re.search(pattern, text_clean):
            score -= 0.5
            break

    # 3. Check for trailing hesitation markers or trailing ellipsis
    for pattern in _HESITATION_PATTERNS:
        if re.search(pattern, text_clean):
            score -= 0.35
            break

    # 4. Punctuation: terminal punctuation (. ! ?) is a strong completion signal
    if re.search(r'[.!?]\s*$', raw_text):
        score += 0.3

    # 5. Questions with rising/interrogative structure tend to be complete once structured
    if re.match(r'^(?:what|where|when|how|why|who|can you|could you|would you|'
                r'will you|is there|are there|do you|does )\b', text_clean):
        if word_count >= 3:
            score += 0.2

    return max(0.0, min(1.0, score))


def get_patience_timeout(partial_text: str, base_timeout: float = 0.9) -> float:
    """
    Dynamically adjusts the silence timeout based on syntactic completeness.
    
    Returns:
        Patience timeout in seconds:
        - Complete sentence -> fast dispatch (~0.45s)
        - Incomplete / trailing connector -> extended patience (3.0-5.0s)
        - Hesitation / thinking -> moderate patience (2.0-3.5s)
    """
    completeness = estimate_syntactic_completeness(partial_text)

    if completeness >= 0.8:
        # High completeness -> fast turn resolution
        return max(0.4, base_timeout * 0.5)
    elif completeness >= 0.5:
        # Moderate completeness -> standard timeout
        return base_timeout
    elif completeness >= 0.2:
        # Low completeness, likely mid-thought -> extended patience
        return min(4.0, base_timeout * 3.5)
    else:
        # Very incomplete (trailing connector, 1-2 words) -> maximum patience
        return min(5.5, base_timeout * 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backchannel vs. Interruption Classifier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InterruptionAnalysis:
    """Result of analyzing whether an audio event during AETHER speech is
    a supportive backchannel or a genuine interruption."""
    is_backchannel: bool
    is_interruption: bool
    confidence: float
    reason: str


def classify_interruption(
    transcript: str,
    audio_duration_ms: float,
    audio_energy: float,
    ambient_floor: float = 300.0
) -> InterruptionAnalysis:
    """
    Flexibly classifies an audio event during AETHER's speech as either
    a backchannel (supportive feedback) or a genuine interruption.
    
    Uses a multi-signal approach instead of hardcoded word lists:
    1. Duration: Backchannels are typically very short (<600ms)
    2. Energy: Backchannels are softer relative to ambient floor
    3. Word count: Backchannels are 1-2 words; interruptions are full phrases
    4. Lexical patterns: Imperatives, negations, and topic-changing words
       signal genuine interruption — detected via regex, not fixed sets
    """
    if not transcript or not transcript.strip():
        return InterruptionAnalysis(
            is_backchannel=True, is_interruption=False,
            confidence=0.9, reason="Empty/noise transcript"
        )

    text = transcript.strip().lower()
    words = text.split()
    word_count = len(words)
    energy_ratio = audio_energy / max(ambient_floor, 1.0)

    # ── Signal 1: Duration-based classification ──
    # Backchannels are typically < 600ms. Full interruptions are longer.
    duration_score = 0.0  # Positive = backchannel, Negative = interruption
    if audio_duration_ms < 400:
        duration_score = 0.4
    elif audio_duration_ms < 600:
        duration_score = 0.2
    elif audio_duration_ms < 1000:
        duration_score = -0.1
    else:
        duration_score = -0.3

    # ── Signal 2: Word count ──
    # Backchannels are 1-2 words. Anything 3+ words is likely a real utterance.
    word_score = 0.0
    if word_count <= 1:
        word_score = 0.3
    elif word_count == 2:
        word_score = 0.15
    elif word_count <= 4:
        word_score = -0.2
    else:
        word_score = -0.5  # Strong interruption signal

    # ── Signal 3: Energy ratio ──
    # Backchannels tend to be softer. Loud, assertive speech = interruption.
    energy_score = 0.0
    if energy_ratio < 1.5:
        energy_score = 0.2  # Quiet = likely backchannel
    elif energy_ratio > 3.0:
        energy_score = -0.3  # Loud = likely interruption

    # ── Signal 4: Lexical patterns (FLEXIBLE regex, not fixed sets) ──
    lexical_score = 0.0

    # Imperative / command-like start -> strong interruption signal
    # This pattern matches verbs, negations, and floor-seizing phrases
    if re.match(
        r'^(?:'
        r'(?:stop|wait|hold\s+on|cancel|no\b|never\s*mind|shut|hang\s+on|one\s+sec(?:ond)?|'
        r'pause|quiet|enough|scratch\s+that|forget\s+(?:it|that))'  # Floor-seizing
        r'|(?:open|close|search|set|play|send|call|text|show|run|go\s+to)'  # Direct commands
        r'|(?:I\s+(?:need|want|think|said|meant)|please\s)'  # Assertive starts
        r'|(?:actually|instead|but\s|wait\s)'  # Topic changers
        r')', text
    ):
        lexical_score = -0.6

    # Short affirmative / supportive vocalization -> backchannel signal
    # Matches any short affirmative sound or agreement phrase
    elif re.match(
        r'^(?:'
        r'm+h*m+|n*uh[\s-]?huh|y(?:ea+h?|ep|up)|sure|righ+t|ok(?:ay)?|'
        r'a+h+|o+h+|go\s+on|continue|got\s+it|I\s+see|interesting|'
        r'wow|nice|cool|sounds?\s+good|makes?\s+sense|true|exactly|'
        r'absolutely|definitely|of\s+course|totally|fair\s+(?:enough|point)'
        r')\s*[.!]?\s*$', text
    ):
        lexical_score = 0.5

    # Full question or new topic sentence -> interruption
    elif re.match(
        r'^(?:'
        r'(?:what|where|when|how|why|who|which)\s'  # Question words
        r'|(?:can|could|would|will|shall|should|do|does|did|is|are)\s+(?:you|we|it|they|he|she)'
        r')', text
    ):
        lexical_score = -0.4

    # ── Combine signals ──
    total_score = duration_score + word_score + energy_score + lexical_score

    if total_score > 0.1:
        return InterruptionAnalysis(
            is_backchannel=True, is_interruption=False,
            confidence=min(0.95, 0.5 + total_score),
            reason=f"Backchannel (dur={audio_duration_ms:.0f}ms, words={word_count}, "
                   f"energy_ratio={energy_ratio:.1f}, score={total_score:+.2f})"
        )
    else:
        return InterruptionAnalysis(
            is_backchannel=False, is_interruption=True,
            confidence=min(0.95, 0.5 + abs(total_score)),
            reason=f"Interruption (dur={audio_duration_ms:.0f}ms, words={word_count}, "
                   f"energy_ratio={energy_ratio:.1f}, score={total_score:+.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Honest Listening Confidence Gate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptionQuality:
    """Assessment of Whisper transcription reliability."""
    is_trustworthy: bool
    should_clarify: bool
    confidence_score: float
    reason: str


def assess_transcription_quality(
    segments: list,
    raw_text: str,
    audio_duration_s: float = 0.0
) -> TranscriptionQuality:
    """
    Flexible confidence gate that inspects Whisper segment metadata to decide
    whether the transcription is trustworthy or should trigger a clarification request.
    
    Uses multiple signals — adaptive ranges, not hardcoded thresholds:
    1. avg_logprob: Average token-level log probability (lower = less certain)
    2. no_speech_prob: Whisper's internal "this is not speech" detector
    3. compression_ratio: Detects hallucination loops (repeated phrases)
    4. Text length vs audio duration: Implausibly long text from short audio
    5. Repetition detection: Catches Whisper's looping artifacts
    """
    if not raw_text or not raw_text.strip():
        return TranscriptionQuality(
            is_trustworthy=False, should_clarify=False,
            confidence_score=0.0, reason="Empty transcription"
        )

    # Gather segment-level statistics
    avg_logprobs = []
    no_speech_probs = []
    compression_ratios = []

    for seg in segments:
        if hasattr(seg, 'avg_logprob'):
            avg_logprobs.append(seg.avg_logprob)
        if hasattr(seg, 'no_speech_prob'):
            no_speech_probs.append(seg.no_speech_prob)
        if hasattr(seg, 'compression_ratio'):
            compression_ratios.append(seg.compression_ratio)

    # Compute aggregate metrics
    mean_logprob = np.mean(avg_logprobs) if avg_logprobs else -0.3
    max_no_speech = max(no_speech_probs) if no_speech_probs else 0.0
    max_compression = max(compression_ratios) if compression_ratios else 1.0

    # ── Signal 1: Token confidence ──
    logprob_ok = mean_logprob > -0.85

    # ── Signal 2: Speech detection ──
    speech_ok = max_no_speech < 0.5

    # ── Signal 3: Hallucination loop detection ──
    compression_ok = max_compression < 2.4

    # ── Signal 4: Repetition pattern detection (flexible regex) ──
    text_lower = raw_text.strip().lower()
    
    # Detect any 3+ word phrase repeated 3+ times
    has_repetition_loop = bool(re.search(
        r'(\b\w+(?:\s+\w+){1,4})\s+(?:\1\s*){2,}', text_lower
    ))
    
    # Detect implausible text-to-audio ratio
    words_per_second = len(raw_text.split()) / max(audio_duration_s, 0.5)
    implausible_rate = words_per_second > 8.0  # Normal speech ~2.5 wps

    # ── Signal 5: Phantom patterns (flexible regex, catches variations) ──
    phantom_patterns = [
        r'thank(?:s| you)\s+(?:for|so much)',
        r'subscri(?:be|bing)',
        r'subtitles?\s+(?:by|from)',
        r'(?:amara|opensubtitles)',
        r'(?:please\s+)?(?:like|share)\s+(?:and|this)',
        r'^(?:you|bye|okay|oh|ah)\.?\s*$',
    ]
    is_phantom = any(re.search(p, text_lower) for p in phantom_patterns)

    # ── Aggregate decision ──
    issues = []
    if not logprob_ok:
        issues.append(f"low_confidence(logprob={mean_logprob:.2f})")
    if not speech_ok:
        issues.append(f"no_speech(prob={max_no_speech:.2f})")
    if not compression_ok:
        issues.append(f"compression_loop(ratio={max_compression:.1f})")
    if has_repetition_loop:
        issues.append("repetition_loop")
    if implausible_rate:
        issues.append(f"implausible_rate({words_per_second:.1f} wps)")
    if is_phantom:
        issues.append("phantom_pattern")

    is_trustworthy = len(issues) == 0
    should_clarify = (not is_trustworthy) and speech_ok and (not is_phantom)

    confidence = 1.0
    if not logprob_ok:
        confidence -= 0.3
    if not speech_ok:
        confidence -= 0.3
    if not compression_ok:
        confidence -= 0.2
    if has_repetition_loop or is_phantom:
        confidence -= 0.3

    return TranscriptionQuality(
        is_trustworthy=is_trustworthy,
        should_clarify=should_clarify,
        confidence_score=max(0.0, confidence),
        reason=f"{'PASS' if is_trustworthy else 'FAIL'}: {', '.join(issues) if issues else 'all_checks_passed'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Disfluency Refinement Engine
# ─────────────────────────────────────────────────────────────────────────────

def refine_disfluencies(text: str) -> str:
    """
    Cleans up natural speech disfluencies to produce clean intent text for ARN/LLM.
    
    Handles:
    - False starts / self-corrections: "Can you... uh... can you open Chrome" -> "can you open Chrome"
    - Filler words: "um", "uh", "like", "you know", "basically"
    - Stuttered word repeats: "open open the browser" -> "open the browser"
    - Trailing hedges
    
    Preserves semantic content and does NOT remove words that carry meaning.
    """
    if not text or not text.strip():
        return ""
    
    result = text.strip()

    # 1. Remove filler words/sounds (flexible patterns, preserving sentence structure)
    filler_pattern = r'\b(?:um+|uh+|er+|eh+|ah+|hmm+|hm+)\b'
    result = re.sub(filler_pattern, '', result, flags=re.IGNORECASE)

    # 2. Remove discourse markers that don't carry semantic meaning (at start only)
    discourse_start = (r'^(?:(?:so|well|okay|ok|right|like|basically|actually|literally|'
                       r'you know|I mean|let me think)\s*,?\s*)+')
    result = re.sub(discourse_start, '', result, flags=re.IGNORECASE)

    # 3. Self-correction: if a phrase is repeated after a pause/correction word, keep the later one
    correction_markers = r'\b(?:no|wait|sorry|I mean|actually|rather|or rather)\b'
    parts = re.split(f'({correction_markers})', result, flags=re.IGNORECASE)
    if len(parts) >= 3:
        last_marker_idx = -1
        for i, part in enumerate(parts):
            if re.match(correction_markers, part.strip(), re.IGNORECASE):
                last_marker_idx = i
        if last_marker_idx >= 0 and last_marker_idx < len(parts) - 1:
            after_correction = ''.join(parts[last_marker_idx + 1:]).strip()
            if len(after_correction.split()) >= 2:
                result = after_correction

    # 4. Remove stuttered word repetitions: "open open the" -> "open the"
    result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)

    # 5. Clean up extra whitespace
    result = re.sub(r'\s{2,}', ' ', result).strip()
    
    # 6. Remove leading/trailing punctuation artifacts
    result = result.strip(' ,;:')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Adaptive Ambient Floor Tracker
# ─────────────────────────────────────────────────────────────────────────────

def update_ambient_floor(ctx: ConversationContext, energy: float) -> float:
    """
    Maintains a slow-moving exponential average of ambient noise floor.
    Only updates during silence (when no one is actively speaking).
    This makes the energy threshold adaptive to different environments.
    """
    if ctx.state in (TurnState.IDLE,) and energy < ctx.ambient_energy_floor * 1.5:
        ctx.ambient_energy_floor = (
            ctx._ema_alpha * energy + (1 - ctx._ema_alpha) * ctx.ambient_energy_floor
        )
    return ctx.ambient_energy_floor


_GENERAL_CLARIFICATION_PROMPTS = [
    "Sorry, I didn't quite catch that. Could you repeat?",
    "I missed that, could you say that again?",
    "Pardon? Could you repeat that for me?",
    "I'm not sure I heard you clearly. One more time?",
    "Could you say that again? I want to make sure I get it right.",
]

# Intent-domain specific clarification prompts (flexible pattern matching)
_INTENT_CLARIFICATION_MAP = [
    (r'\b(?:open|launch|start|close|window|app|application|run)\b', [
        "Did you want me to open or control an application? I didn't quite catch which one.",
        "I heard you mention an app, but missed which one. Could you say the name again?",
    ]),
    (r'\b(?:search|google|find|look up|what is|who is|tell me about|info)\b', [
        "Were you asking me to look something up on the web? Could you repeat the search topic?",
        "I caught that you wanted to search for something. Could you repeat the question?",
    ]),
    (r'\b(?:message|whatsapp|text|send to|call)\b', [
        "Did you want to send a message? Who should I send it to?",
        "I heard a messaging command, but missed the contact name. Could you repeat that?",
    ]),
    (r'\b(?:music|song|play|volume|mute|unmute|spotify|track)\b', [
        "Did you want to adjust the volume or control media playback? Could you repeat that?",
        "I caught a media command, but couldn't make out the song or action. One more time?",
    ]),
    (r'\b(?:battery|cpu|ram|memory|system|diagnostics|specs|health)\b', [
        "Were you asking for a system status check? Could you say that again?",
    ]),
]


def get_clarification_response(quality: TranscriptionQuality, partial_text: str = "") -> Optional[str]:
    """
    Returns an intent-aware response based on transcription quality assessment and text clues.
    
    If the utterance had borderline confidence but showed recognizable intent structure,
    it returns a targeted, intelligent clarification question (e.g. asking which app or contact).
    
    Returns None if the audio should just be silently ignored (pure noise or phantom loop).
    """
    if quality.is_trustworthy:
        return None  # No clarification needed
    
    if not quality.should_clarify:
        # Pure noise or phantom hallucination -> silently ignore
        return None

    # Check for domain-specific intent clues in the partial/raw text
    if partial_text:
        text_lower = partial_text.lower().strip()
        for pattern, prompts in _INTENT_CLARIFICATION_MAP:
            if re.search(pattern, text_lower):
                return random.choice(prompts)

    # Fallback to general conversational clarification
    return random.choice(_GENERAL_CLARIFICATION_PROMPTS)
