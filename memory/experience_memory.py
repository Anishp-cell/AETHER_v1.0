import json
import os
import threading
import requests
import re

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")

# Facts that are ALWAYS noise — the LLM makes these up when it has nothing real to extract
HALLUCINATION_KEYWORDS = [
    "favorite color", "favourite color",
    "User's name is",
    "I am unsure",
    "I don't know",
    "I cannot",
    "I'm not sure",
    "think I am",       # Echoed questions like "what kind of person I am"
    "you think I",
]

# The user must use one of these possessive patterns for it to be a real personal fact
# e.g. "my name", "I am", "I love", "I hate", "I work", "I prefer"
PERSONAL_SIGNAL_PATTERNS = [
    r"\bmy\b", r"\bi am\b", r"\bi'm\b", r"\bi love\b", r"\bi hate\b",
    r"\bi prefer\b", r"\bi like\b", r"\bi enjoy\b", r"\bi work\b",
    r"\bi study\b", r"\bi always\b", r"\bi never\b", r"\bcall me\b",
    r"\bmy name\b", r"\bmy goal\b", r"\bmy favorite\b", r"\bmy favourite\b",
]

class ExperienceMemory:
    def __init__(self, llm_url="http://127.0.0.1:11434/api/generate", base_model="qwen2.5:1.5b"):
        self.profile_path = PROFILE_PATH
        self.llm_url = llm_url
        self.base_model = base_model
        
        if not os.path.exists(self.profile_path):
            self.save_profile({"Facts": []})

    def load_profile(self) -> dict:
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"Facts": []}

    def save_profile(self, data: dict):
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _has_personal_signal(self, text: str) -> bool:
        """
        Pre-flight check: Does the user's raw speech actually contain
        a first-person DECLARATION (not a question) with a personal pattern?
        Questions like 'What kind of person am I?' are correctly rejected.
        """
        # Reject pure questions — they cannot contain personal declarations
        stripped = text.strip()
        if stripped.endswith("?"):
            return False
        # Reject if the ONLY 'I am' pattern is buried inside a question clause
        text_lower = stripped.lower()
        # Must have personal signal AND not be entirely a question
        return any(re.search(pat, text_lower) for pat in PERSONAL_SIGNAL_PATTERNS)

    def _is_hallucinated(self, extracted: str) -> bool:
        """Catches common LLM fabrications."""
        lower = extracted.lower()
        return any(kw.lower() in lower for kw in HALLUCINATION_KEYWORDS)

    def extract_and_store_async(self, user_statement: str):
        """
        Two-gate system:
        Gate 1 (instant, zero-cost): Does the statement even contain a personal signal?
        Gate 2 (LLM call):           Does the LLM confirm it's a permanent fact?
                                     Must pass a verification check to be stored.
        """
        def background_extraction():
            try:
                # ── GATE 1: Pre-flight personal signal check ──
                if not self._has_personal_signal(user_statement):
                    print(f"[Deep Memory] No personal signal detected — skipping LLM call.")
                    return

                # ── GATE 2: Two-step LLM extraction + verification ──
                prompt = f"""You are a strict personal fact extractor. Your ONLY job is to identify permanent personal facts that the user reveals about themselves.

User said: "{user_statement}"

Rules:
- A personal fact must be something the user explicitly stated about themselves using words like "I", "my", "me", "I am", "I love", "I hate", "my name", "my goal", etc.
- Do NOT infer, assume, or hallucinate facts not present in the text.
- Do NOT extract temporary states like "I'm tired right now" or "I feel cold" — these are not permanent.
- Permanent facts include: name, profession, hobbies, preferences, long-term goals, relationships.

If a clear permanent personal fact exists: return ONLY the fact as one concise sentence. Example: "User loves hiking."
If no permanent fact exists: return exactly "NONE"."""
                
                payload = {
                    "model": self.base_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}  # Deterministic
                }
                
                res = requests.post(self.llm_url, json=payload, timeout=10).json()
                extracted = res.get("response", "").strip()
                # Remove any leading/trailing quotes the model might add
                extracted = extracted.strip('"\'')
                
                print(f"[Deep Memory Debug] LLM Extracted: '{extracted}'")

                # ── GATE 2b: Anti-hallucination filter ──
                if not extracted or "NONE" in extracted.upper():
                    return
                if len(extracted) > 200:
                    print(f"[Deep Memory] Response too long ({len(extracted)} chars) — discarding.")
                    return
                if self._is_hallucinated(extracted):
                    print(f"[Deep Memory] ⚠️ Hallucination detected — discarding: '{extracted}'")
                    return

                # ── STORE ──
                profile = self.load_profile()
                # Avoid near-duplicate facts
                existing = [f.lower() for f in profile.get("Facts", [])]
                if extracted.lower() not in existing:
                    profile["Facts"].append(extracted)
                    self.save_profile(profile)
                    print(f"\n🧠 [Deep Memory] Stored: '{extracted}'")
                else:
                    print(f"[Deep Memory] Duplicate skipped: '{extracted}'")
                    
            except Exception as e:
                print(f"[Deep Memory Error] {e}")
                
        thread = threading.Thread(target=background_extraction, daemon=True)
        thread.start()

    def forget_fact(self, keyword: str):
        """Remove facts containing a keyword. E.g. forget_fact('sleep') removes sleep-related facts."""
        profile = self.load_profile()
        before = len(profile["Facts"])
        profile["Facts"] = [f for f in profile["Facts"] if keyword.lower() not in f.lower()]
        after = len(profile["Facts"])
        self.save_profile(profile)
        print(f"[Deep Memory] Removed {before - after} fact(s) matching '{keyword}'.")
        return before - after
