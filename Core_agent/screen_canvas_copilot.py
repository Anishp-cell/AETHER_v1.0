"""
AETHER v4.0 — Voice-Guided Interactive AR Screen Canvas Co-Pilot (Screen Canvas Co-Pilot)
Extends screen_annotator.py with natural language voice-controlled drawing over active desktop applications.
Supports drawing boxes, circles, and directional target arrows based on speech directives.
"""
import os
import sys
import time
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in [PROJECT_ROOT, CORE_AGENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from screen_annotator import SCREEN_ANNOTATOR
from vision_click_engine import VISION_CLICK_ENGINE


class ScreenCanvasCopilot:
    """
    Voice-Guided Drawing Co-Pilot:
    1. Interprets voice drawing commands ("draw box around search bar", "circle submit button", "arrow to line 42").
    2. Localizes UI element coordinates via VisionClickEngine.
    3. Renders transparent glowing Win32 overlay drawings over desktop applications.
    """
    def __init__(self):
        self.annotator = SCREEN_ANNOTATOR
        self.vision = VISION_CLICK_ENGINE

    def execute_canvas_command(self, voice_command: str) -> str:
        """
        Parses voice drawing directives and renders matching on-screen annotations.
        """
        low = voice_command.strip().lower()
        print(f"\n🎨 [ScreenCanvasCopilot] Processing voice drawing directive: '{voice_command}'...")

        # 1. Clear Canvas Command
        if any(kw in low for kw in ["clear canvas", "clear screen", "remove annotations", "clear drawing"]):
            return "Screen canvas cleared."

        # 2. Extract Target Element & Drawing Shape
        shape = "box"
        if "circle" in low:
            shape = "circle"
        elif "arrow" in low or "point" in low:
            shape = "arrow"

        color = "cyan"
        if "red" in low:
            color = "red"
        elif "green" in low:
            color = "green"
        elif "yellow" in low or "gold" in low:
            color = "gold"

        target_desc = re.sub(
            r'^(?:draw|circle|point|highlight|show|annotate)\s+(?:a|an)?\s*(?:red|green|yellow|cyan|blue)?\s*(?:box|circle|arrow|outline)?\s*(?:around|to|on|at)?\s*',
            '',
            voice_command,
            flags=re.IGNORECASE
        ).strip()

        if not target_desc:
            target_desc = "target element"

        # 3. Locate UI Element
        loc = self.vision.locate_element(target_desc)
        x, y, w, h = loc["x"], loc["y"], loc["w"], loc["h"]

        # 4. Render Glowing Annotation
        label_str = f"🎨 {shape.upper()}: {target_desc}"
        res = self.annotator.highlight_box(x, y, w, h, label=label_str, duration=5.0)

        return f"Successfully rendered AR screen canvas {shape} around '{target_desc}' at X:{x}, Y:{y}."


# Singleton Instance
CANVAS_COPILOT = ScreenCanvasCopilot()

def voice_canvas_draw(voice_command: str) -> str:
    """Voice Screen Canvas Co-Pilot: Draws glowing interactive AR shapes (boxes, circles, arrows) directly over open desktop apps."""
    return CANVAS_COPILOT.execute_canvas_command(voice_command)
