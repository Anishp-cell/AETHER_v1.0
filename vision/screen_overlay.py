"""
AETHER Vision Module — Screen Border Overlay
Creates a native Windows transparent top-most animated overlay with a glowing orange border
when AETHER captures the screen, providing instant visual confirmation in <35ms.
"""
import tkinter as tk
import threading
import time

def flash_orange_border(duration_ms=350, border_thickness=8, color="#FF8C00"):
    """
    Spawns an asynchronous top-most transparent window with an animated glowing orange border
    emerging from edge centers towards the screen vertices.
    """
    def _overlay_thread():
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.wm_attributes("-topmost", True)
            
            # Use chroma key for 100% background transparency
            transparent_key = "#000001"
            root.config(bg=transparent_key)
            try:
                root.wm_attributes("-transparentcolor", transparent_key)
            except Exception:
                pass

            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.geometry(f"{screen_w}x{screen_h}+0+0")

            canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg=transparent_key, highlightthickness=0)
            canvas.pack(fill="both", expand=True)

            cx = screen_w / 2
            cy = screen_h / 2

            steps = 10
            delay_per_step = (duration_ms * 0.6) / steps / 1000.0

            def draw_step(progress):
                canvas.delete("border")
                w_offset = (screen_w / 2) * progress
                h_offset = (screen_h / 2) * progress

                # Top & Bottom horizontal expanding lines
                canvas.create_line(cx - w_offset, border_thickness/2, cx + w_offset, border_thickness/2, fill=color, width=border_thickness, tags="border")
                canvas.create_line(cx - w_offset, screen_h - border_thickness/2, cx + w_offset, screen_h - border_thickness/2, fill=color, width=border_thickness, tags="border")

                # Left & Right vertical expanding lines
                canvas.create_line(border_thickness/2, cy - h_offset, border_thickness/2, cy + h_offset, fill=color, width=border_thickness, tags="border")
                canvas.create_line(screen_w - border_thickness/2, cy - h_offset, screen_w - border_thickness/2, cy + h_offset, fill=color, width=border_thickness, tags="border")
                root.update_idletasks()

            # Phase 1: Expanding Border Animation
            for i in range(1, steps + 1):
                draw_step(i / steps)
                time.sleep(delay_per_step)

            # Phase 2: Full Border Flash
            canvas.delete("border")
            canvas.create_rectangle(
                border_thickness/2, border_thickness/2, 
                screen_w - border_thickness/2, screen_h - border_thickness/2, 
                outline="#FFA500", width=border_thickness, tags="border"
            )
            root.update_idletasks()
            time.sleep(0.12)

            root.destroy()
        except Exception as e:
            print(f"[Screen Overlay Warning] Could not render overlay: {e}")

    thread = threading.Thread(target=_overlay_thread, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    print("Testing Animated Orange Screen Border Overlay...")
    t = flash_orange_border(duration_ms=400)
    time.sleep(1)
    print("Test complete.")
