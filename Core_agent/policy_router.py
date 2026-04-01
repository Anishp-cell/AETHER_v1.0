"""
AETHER - Policy Router (Phase 6, Patent Claims 1, 5, 7)
The central authority for policy-gated expert activation.
Takes Micro-Expert signals as input, applies hardcoded rule logic,
and dynamically composes a System Prompt from the base persona + selected Behavioral Adapters.

This is NOT a learned gating mechanism. It operates on strict IF/THEN rules (Claim 7).
"""

# ── BEHAVIORAL ADAPTERS (Patent Claim 5) ──────────────────────────────────────
# Each adapter is a string fragment injected into the system prompt.
# Multiple adapters can be composed together (Claim 5: "multiple adapters may be combined").

ADAPTER_CONCISE = """
[Adapter: Precision Mode] The user needs a fast, direct answer. Strip all filler.
Be ruthlessly brief. One or two sentences max. No pleasantries."""

ADAPTER_DEBUGGING = """
[Adapter: Senior Debugger Mode] The user is experiencing a technical problem.
Think systematically. Ask ONE clarifying question if needed. Suggest the most likely root cause first."""

ADAPTER_CREATIVE = """
[Adapter: Creative Engine Mode] The user wants something creative.
Be expressive, imaginative, and surprising. Use vivid language. Enjoy it."""

ADAPTER_FRUSTRATED_USER = """
[Adapter: Empathy Override] The user sounds frustrated.
Acknowledge this immediately and briefly. Then solve the problem fast and cleanly.
No unnecessary preamble. Show you understand and are on it."""

ADAPTER_CASUAL = """
[Adapter: Casual Companion Mode] This is light conversation.
Be warm, witty, and playful. Throw in a touch of dry British humour when appropriate.
You're not a robot reading answers - you're a sharp, charming AI who actually enjoys talking."""

ADAPTER_EXCITED = """
[Adapter: High-Energy Mode] The user is enthusiastic!
Match their energy. Be snappy, upbeat, and fun. Celebrate with them."""

ADAPTER_SEARCH = """
[Adapter: Research Mode] Invoking AetherWeb to gather external data.
When AetherWeb returns results, synthesize them into a confident, natural spoken summary.
Don't just list facts - tell a short story with them."""

ADAPTER_URGENT = """
[Adapter: Emergency Protocol] The user needs something done right now.
Skip all preamble. Execute the task immediately and confirm completion in one sentence."""


class PolicyRouter:
    """
    The central rule-based decision authority (Patent Claims 1 & 7).
    Selects and composes Behavioral Adapters based on Micro-Expert signals.
    """

    def route(self, signals: dict, base_prompt: str) -> tuple[str, str]:
        """
        Takes the Micro-Expert signals and base system prompt,
        applies policy rules, and returns:
        - (composed_system_prompt, routing_log)
        """
        intent = signals.get("intent", "casual conversation")
        emotion = signals.get("emotion", "neutral")
        confidence = signals.get("intent_confidence", 0.5)

        adapters = []
        routing_log = []

        # ── POLICY RULES (Claim 7: governed by policy constraints) ──

        # Rule 1: Emotion gates take priority
        if emotion == "frustrated":
            adapters.append(ADAPTER_FRUSTRATED_USER)
            adapters.append(ADAPTER_CONCISE)
            routing_log.append("Emotion:Frustrated → [Empathy + Precision]")

        elif emotion == "urgent":
            adapters.append(ADAPTER_URGENT)
            routing_log.append("Emotion:Urgent → [Emergency Protocol]")

        elif emotion == "excited":
            adapters.append(ADAPTER_EXCITED)
            routing_log.append("Emotion:Excited → [High-Energy]")

        # Rule 2: Intent-based expert activation
        if "debug" in intent or "debugging" in intent:
            adapters.append(ADAPTER_DEBUGGING)
            routing_log.append("Intent:Debugging → [Senior Debugger]")

        elif "creative" in intent:
            adapters.append(ADAPTER_CREATIVE)
            routing_log.append("Intent:Creative → [Creative Engine]")

        elif "web search" in intent or "knowledge" in intent:
            adapters.append(ADAPTER_SEARCH)
            routing_log.append("Intent:Knowledge → [Research Mode]")

        elif "casual" in intent and emotion == "neutral":
            adapters.append(ADAPTER_CASUAL)
            routing_log.append("Intent:Casual → [Companion Mode]")

        # Rule 3: Low-confidence fallback (for short/ambiguous inputs)
        if confidence < 0.45 and not adapters:
            adapters.append(ADAPTER_CASUAL)
            routing_log.append(f"LowConfidence({confidence:.0%}) → [Companion Mode]")

        # ── COMPOSE FINAL PROMPT ──
        if adapters:
            adapter_block = "\n".join(adapters)
            composed_prompt = f"{base_prompt}\n\n{adapter_block}"
        else:
            composed_prompt = base_prompt

        log_str = " | ".join(routing_log) if routing_log else "Default"
        print(f"🔀 [Policy Router] Rules fired: {log_str}")

        return composed_prompt, log_str
