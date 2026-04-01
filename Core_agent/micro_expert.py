"""
AETHER - Micro-Expert Module (Phase 6, Patent Claim 6)
A lightweight, non-authoritative behavioral signal generator.
Uses zero-shot NLP classification to output Intent, Emotion, and Routing Confidence
signals BEFORE the base LLM even processes the query.

This model does NOT generate language. It does NOT make decisions.
Its outputs are advisory-only inputs for the Policy Router.
"""
import threading
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Intent categories (Claim 7: policy-routed expert activation)
INTENT_LABELS = [
    "desktop command",       # Open app, type, press key, run program
    "web search",            # Look something up silently on the web
    "knowledge question",    # Factual or educational question
    "creative request",      # Write poem, story, code, song etc.
    "screen analysis",       # What's on the screen?
    "casual conversation",   # Small talk / chitchat
    "debugging",             # Fix an error, read an exception
]

# Emotion categories
EMOTION_LABELS = [
    "neutral",
    "frustrated",
    "excited",
    "urgent",
]

class MicroExpert:
    """
    ~20M parameter classifier engine that produces non-authoritative
    behavioral signals as described in Patent Claim 6.
    """

    def __init__(self):
        print("[Micro-Expert] Loading lightweight NLP signal classifier...")
        self._classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=-1  # CPU only — tiny auxiliary model (Claim 6)
        )
        self._cache: dict[str, dict] = {}  # Phase 7: result cache
        print("[Micro-Expert] Signal classifier online.")

    def analyze(self, text: str) -> dict:
        """
        Analyzes the user's raw text and returns a structured signal dict.
        Results are cached so identical inputs skip the NLP pipeline entirely.
        """
        if not text or len(text.strip()) < 3:
            return {"intent": "casual conversation", "intent_confidence": 0.5, "emotion": "neutral"}

        # Phase 7: Cache hit — skip NLP entirely
        cache_key = text.strip().lower()[:120]
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            print(f"\n🧬 [Micro-Expert Cache Hit] Intent: '{cached['intent']}' | Emotion: '{cached['emotion']}'")
            return cached

        try:

            # Run both classifications in parallel threads for speed
            intent_result = [None]
            emotion_result = [None]

            def classify_intent():
                res = self._classifier(text, INTENT_LABELS, multi_label=False)
                intent_result[0] = {"label": res["labels"][0], "score": res["scores"][0]}

            def classify_emotion():
                res = self._classifier(text, EMOTION_LABELS, multi_label=False)
                emotion_result[0] = {"label": res["labels"][0], "score": res["scores"][0]}

            t1 = threading.Thread(target=classify_intent)
            t2 = threading.Thread(target=classify_emotion)
            t1.start(); t2.start()
            t1.join(); t2.join()

            signals = {
                "intent": intent_result[0]["label"],
                "intent_confidence": round(intent_result[0]["score"], 3),
                "emotion": emotion_result[0]["label"],
            }
            print(f"\n🧬 [Micro-Expert Signal] Intent: '{signals['intent']}' ({signals['intent_confidence']:.0%}) | Emotion: '{signals['emotion']}'")
            # Phase 7: Store in cache (cap at 64 entries to avoid memory bloat)
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = signals
            return signals

        except Exception as e:
            print(f"[Micro-Expert Error] {e}")
            return {"intent": "casual conversation", "intent_confidence": 0.5, "emotion": "neutral"}
