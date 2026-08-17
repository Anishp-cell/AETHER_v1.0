import json
import os
import threading
import requests
import re
from typing import List, Dict, Tuple, Optional

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
PERSONAL_SIGNAL_PATTERNS = [
    r"\bmy\b", r"\bi am\b", r"\bi'm\b", r"\bi love\b", r"\bi hate\b",
    r"\bi prefer\b", r"\bi like\b", r"\bi enjoy\b", r"\bi work\b",
    r"\bi study\b", r"\bi always\b", r"\bi never\b", r"\bcall me\b",
    r"\bmy name\b", r"\bmy goal\b", r"\bmy favorite\b", r"\bmy favourite\b",
    r"\bmy birthday\b", r"\bmy phone\b", r"\bmy email\b", r"\bmy address\b",
    r"\bremember that\b", r"\bkeep in mind that\b", r"\bdon't forget that\b"
]

class ExperienceMemory:
    """
    Tier 2 Memory Engine for AETHER:
    - Stores persistent user profile, preferences, and biographical facts in user_profile.json.
    - Supports dual-trigger ingestion: automatic background extraction + explicit voice commands.
    - Provides semantic fact retrieval to condition LLM reasoning prompts.
    """
    def __init__(self, llm_url="http://127.0.0.1:11434/api/generate", base_model="qwen2.5-coder:1.5b"):
        self.profile_path = PROFILE_PATH
        self.llm_url = llm_url
        self.base_model = base_model
        self.lock = threading.Lock()
        
        if not os.path.exists(self.profile_path):
            self.save_profile({
                "name": "User",
                "location": "Pune",
                "reminders": [],
                "remembered_facts": [],
                "preferred_apps": [],
                "last_seen": ""
            })

    def load_profile(self) -> dict:
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalize legacy 'Facts' key to 'remembered_facts'
                if "Facts" in data and "remembered_facts" not in data:
                    data["remembered_facts"] = data.pop("Facts")
                elif "Facts" in data and isinstance(data["Facts"], list):
                    for f in data.pop("Facts"):
                        if f not in data.get("remembered_facts", []):
                            data.setdefault("remembered_facts", []).append(f)
                return data
        except Exception as e:
            print(f"[ExperienceMemory Warning] Could not load profile: {e}")
            return {
                "name": "User",
                "location": "",
                "reminders": [],
                "remembered_facts": [],
                "preferred_apps": [],
                "last_seen": ""
            }

    def save_profile(self, data: dict):
        with self.lock:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

    def get_all_facts(self) -> List[str]:
        """Returns all stored persistent facts."""
        profile = self.load_profile()
        return profile.get("remembered_facts", [])

    def get_relevant_facts(self, query: str, limit: int = 5) -> List[str]:
        """
        Retrieves user profile facts relevant to the current user query.
        Uses token/keyword overlap scoring for ultra-fast sub-1ms CPU ranking.
        """
        facts = self.get_all_facts()
        if not facts:
            return []
            
        q_tokens = set(re.findall(r'\w+', query.lower()))
        # Remove ultra-common stop words
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "do", "you", "i", "me", "my", "to", "in", "for", "and", "of"}
        q_tokens = q_tokens - stop_words
        
        if not q_tokens:
            return facts[:limit]
            
        scored = []
        for fact in facts:
            f_tokens = set(re.findall(r'\w+', fact.lower()))
            overlap = len(q_tokens & f_tokens)
            scored.append((overlap, fact))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        # Include matching facts or top baseline facts
        matched = [f for score, f in scored if score > 0]
        if matched:
            return matched[:limit]
        return facts[:limit]

    def _has_personal_signal(self, text: str) -> bool:
        """
        Pre-flight check: Does the user's raw speech contain a personal pattern?
        Questions like 'What kind of person am I?' without explicit command are rejected.
        """
        stripped = text.strip()
        text_lower = stripped.lower()
        
        # Explicit memory instructions bypass question filter
        if any(cmd in text_lower for cmd in ["remember that", "don't forget that", "keep in mind", "save note"]):
            return True
            
        if stripped.endswith("?"):
            return False
            
        return any(re.search(pat, text_lower) for pat in PERSONAL_SIGNAL_PATTERNS)

    def _is_hallucinated(self, extracted: str) -> bool:
        """Catches common LLM fabrications."""
        lower = extracted.lower()
        return any(kw.lower() in lower for kw in HALLUCINATION_KEYWORDS)

    def handle_explicit_memory_command(self, user_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects and executes explicit voice/chat memory commands:
        1. 'remember that [fact]' / 'save note [fact]' -> stores fact immediately
        2. 'what do you remember about me' / 'show memories' -> returns known facts
        3. 'forget [keyword]' / 'delete memory about [keyword]' -> removes facts
        Returns: (is_handled, response_string)
        """
        text = user_text.strip()
        lower = text.lower()

        # ── 1. QUERY MEMORY: "What do you remember about me?" ──
        if any(phrase in lower for phrase in [
            "what do you remember about me",
            "what do you know about me",
            "show my memories",
            "show remembered facts",
            "list my preferences",
            "what are my preferences",
            "what have you learned about me"
        ]):
            facts = self.get_all_facts()
            if not facts:
                return (True, "I haven't stored any personal memories about you yet. You can say 'Remember that...' anytime.")
            formatted = "\n".join([f"- {f}" for f in facts])
            return (True, f"Here is what I currently remember about you:\n{formatted}")

        # ── 2. FORGET FACT: "Forget that I like coffee" ──
        forget_match = re.search(r"(?:forget|delete memory about|remove memory about)\s+(?:that\s+)?(.+)", text, re.IGNORECASE)
        if forget_match:
            keyword = forget_match.group(1).strip().strip(".'\"")
            count = self.forget_fact(keyword)
            if count > 0:
                return (True, f"I have forgotten {count} memory record(s) related to '{keyword}'.")
            else:
                return (True, f"I couldn't find any stored memories matching '{keyword}'.")

        # ── 3. EXPLICIT STORE: "Remember that I prefer dark mode" ──
        store_match = re.search(r"(?:remember that|please remember that|keep in mind that|don't forget that|save note that)\s+(.+)", text, re.IGNORECASE)
        if store_match:
            fact_statement = store_match.group(1).strip().strip(".'\"")
            # Directly clean and format statement
            cleaned_fact = fact_statement
            if cleaned_fact.lower().startswith("i "):
                cleaned_fact = "User " + cleaned_fact[2:]
            elif cleaned_fact.lower().startswith("my "):
                cleaned_fact = "User's " + cleaned_fact[3:]
                
            self.store_fact_direct(cleaned_fact)
            return (True, f"I've remembered that: {cleaned_fact}.")

        return (False, None)

    def store_fact_direct(self, fact: str):
        """Directly writes a validated fact into user_profile.json."""
        if not fact or len(fact) < 4:
            return
        fact = fact.strip().strip('"\'')
        profile = self.load_profile()
        facts = profile.get("remembered_facts", [])
        if fact.lower() not in [f.lower() for f in facts]:
            facts.append(fact)
            profile["remembered_facts"] = facts
            self.save_profile(profile)
            print(f"\n[Memory] Direct Stored: '{fact}'")

    def extract_and_store_async(self, user_statement: str):
        """
        Two-gate background fact extraction:
        Gate 1 (instant regex): Does the statement contain personal declarations?
        Gate 2 (deterministic LLM): Extracts permanent biographical facts.
        """
        def background_extraction():
            try:
                if not self._has_personal_signal(user_statement):
                    return

                # Prompt engineering for deterministic personal fact extraction
                prompt = (
                    "You are a strict personal fact extractor. Extract ONLY permanent personal facts stated by the user about themselves.\n\n"
                    f"User said: \"{user_statement}\"\n\n"
                    "Rules:\n"
                    "- Permanent facts include: name, career, programming language, hobbies, preferences, long-term goals, habits.\n"
                    "- Do NOT extract temporary states (e.g. 'I am tired', 'I am hungry').\n"
                    "- If a permanent personal fact exists, return ONLY the fact as one concise third-person sentence (e.g. 'User prefers Python over C++').\n"
                    "- If no permanent fact exists, return exactly 'NONE'."
                )
                
                payload = {
                    "model": self.base_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                
                res = requests.post(self.llm_url, json=payload, timeout=8).json()
                extracted = res.get("response", "").strip().strip('"\'')

                if not extracted or "NONE" in extracted.upper() or len(extracted) > 180 or self._is_hallucinated(extracted):
                    return

                profile = self.load_profile()
                existing = [f.lower() for f in profile.get("remembered_facts", [])]
                if extracted.lower() not in existing:
                    profile.setdefault("remembered_facts", []).append(extracted)
                    self.save_profile(profile)
                    print(f"\n[Memory] Auto-Learned Fact: '{extracted}'")
            except Exception as e:
                pass
                
        thread = threading.Thread(target=background_extraction, daemon=True)
        thread.start()

    def force_store_async(self, user_statement: str):
        """Asynchronously forces extraction of a permanent fact."""
        self.store_fact_direct(user_statement)

    def forget_fact(self, keyword: str) -> int:
        """Removes facts containing keyword."""
        profile = self.load_profile()
        facts = profile.get("remembered_facts", [])
        before = len(facts)
        updated = [f for f in facts if keyword.lower() not in f.lower()]
        after = len(updated)
        profile["remembered_facts"] = updated
        self.save_profile(profile)
        print(f"[ExperienceMemory] Removed {before - after} fact(s) matching '{keyword}'.")
        return before - after
