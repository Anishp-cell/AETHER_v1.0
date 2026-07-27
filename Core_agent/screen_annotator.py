"""
AETHER v2.0 — Screen-Aware Visual Annotation & Drawing Engine
Draws glowing bounding boxes, arrows, and text annotations directly over active desktop windows.
"""
import sys
import os
import time
import threading
import tkinter as tk
import ctypes

class ScreenAnnotator:
    """
    Full-screen transparent overlay for visual screen annotations.
    """
    def __init__(self):
        self.root = None
        self.canvas = None
        self._thread = None
        self.is_active = False

    def highlight_box(self, x: int, y: int, width: int, height: int, label: str = "AETHER Target", duration: float = 4.0):
        """
        Draws a glowing neon cyan/magenta bounding box over screen coordinates (x, y, w, h).
        """
        threading.Thread(target=self._draw_overlay, args=("box", x, y, width, height, label, duration), daemon=True).start()
        return f"Successfully annotated screen box at ({x}, {y}, {width}, {height}) with label '{label}' for {duration}s."

    def draw_arrow(self, x1: int, y1: int, x2: int, y2: int, label: str = "Workflow Step", duration: float = 4.0):
        """
        Draws a glowing directional arrow between (x1, y1) and (x2, y2).
        """
        threading.Thread(target=self._draw_overlay, args=("arrow", x1, y1, x2, y2, label, duration), daemon=True).start()
        return f"Successfully drew screen arrow from ({x1},{y1}) to ({x2},{y2}) with label '{label}' for {duration}s."

    def _draw_overlay(self, mode: str, x1: int, y1: int, x2_or_w: int, y2_or_h: int, label: str, duration: float):
        try:
            root = tk.Tk()
            root.title("AETHER Screen Annotator")
            root.overrideredirect(True)
            root.wm_attributes("-topmost", True)
            
            # Make transparent background
            root.config(bg="#000001")
            root.wm_attributes("-transparentcolor", "#000001")
            
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.geometry(f"{screen_w}x{screen_h}+0+0")
            
            canvas = tk.Canvas(root, bg="#000001", highlightthickness=0)
            canvas.pack(fill="both", expand=True)

            accent_cyan = "#00F5FF"
            accent_magenta = "#FF007F"
            glow_purple = "#A855F7"

            if mode == "box":
                x, y, w, h = x1, y1, x2_or_w, y2_or_h
                # Multi-layer glow bounding box
                canvas.create_rectangle(x - 3, y - 3, x + w + 3, y + h + 3, outline=glow_purple, width=3)
                canvas.create_rectangle(x - 1, y - 1, x + w + 1, y + h + 1, outline=accent_magenta, width=2)
                canvas.create_rectangle(x, y, x + w, y + h, outline=accent_cyan, width=2)
                
                # Corner accents
                corner_len = min(15, w // 4, h // 4)
                canvas.create_line(x, y, x + corner_len, y, fill="#FFFFFF", width=3)
                canvas.create_line(x, y, x, y + corner_len, fill="#FFFFFF", width=3)
                
                # Floating Label Tag
                canvas.create_rectangle(x, y - 24, x + len(label) * 8 + 12, y - 4, fill="#0F172A", outline=accent_cyan, width=1)
                canvas.create_text(x + 6, y - 14, text=f"⚡ {label}", fill=accent_cyan, font=("Consolas", 9, "bold"), anchor="w")

            elif mode == "arrow":
                start_x, start_y, end_x, end_y = x1, y1, x2_or_w, y2_or_h
                # Outer glow arrow
                canvas.create_line(start_x, start_y, end_x, end_y, fill=glow_purple, width=6, arrow=tk.LAST, arrowshape=(16, 20, 8))
                canvas.create_line(start_x, start_y, end_x, end_y, fill=accent_cyan, width=3, arrow=tk.LAST, arrowshape=(14, 18, 6))
                
                # Label text near midpoint
                mid_x = (start_x + end_x) // 2
                mid_y = (start_y + end_y) // 2
                canvas.create_rectangle(mid_x - 4, mid_y - 12, mid_x + len(label) * 8 + 8, mid_y + 8, fill="#0F172A", outline=accent_cyan, width=1)
                canvas.create_text(mid_x + 2, mid_y - 2, text=label, fill="#FFFFFF", font=("Consolas", 8, "bold"), anchor="w")

            root.update()
            time.sleep(duration)
            root.destroy()

        except Exception as e:
            print(f"[Screen Annotator Error] {e}")


# Global Singleton Instance
SCREEN_ANNOTATOR = ScreenAnnotator()

def annotate_screen(x: int = 100, y: int = 100, width: int = 300, height: int = 200, label: str = "Target Button"):
    """
    Visual screen drawing tool: Draws a glowing bounding box over desktop screen coordinates (x, y, width, height) with a label.
    """
    return SCREEN_ANNOTATOR.highlight_box(x, y, width, height, label)
