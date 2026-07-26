"""
AETHER Vision Module — Screen Capture
Captures the primary display as a compressed JPEG base64 string
for injection into multimodal LLM prompts.
"""
import base64
import io
from PIL import Image
import mss

class ScreenCapture:
    def __init__(self):
        # NOTE: We no longer store a persistent mss() instance.
        # mss uses thread-local Windows GDI device contexts (srcdc)
        # that CANNOT be shared across threads. Creating a fresh
        # instance per call avoids the '_thread._local' crash.
        print("[Vision Layer] Screen Capture initialized.")

    def capture(self, max_width=1280, quality=60):
        """
        Captures the primary monitor, resizes to max_width (preserving aspect ratio),
        and returns a base64-encoded JPEG string suitable for LLM vision APIs.
        """
        import time
        try:
            try:
                from vision.screen_overlay import flash_orange_border
            except ImportError:
                from screen_overlay import flash_orange_border
            flash_orange_border(duration_ms=350)
        except Exception as _e:
            print(f"[Vision Layer Warning] Overlay trigger failed: {_e}")


        start_time = time.perf_counter()
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"[Vision Layer] 📸 Screen capture complete in {elapsed_ms:.1f}ms! You can now freely switch windows.")
        
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        
        # Save debug screenshot
        import os
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            debug_dir = os.path.join(base_dir, "debug_screenshots")
            os.makedirs(debug_dir, exist_ok=True)
            img.save(os.path.join(debug_dir, "pre_processing_screen.jpg"), format="JPEG", quality=quality)
            print(f"[Vision Layer] Saved debug screenshot: debug_screenshots/pre_processing_screen.jpg")
        except Exception as e:
            print(f"[Vision Layer] Failed to save debug screenshot: {e}")
        
        # Resize to save bandwidth and LLM token cost
        ratio = max_width / img.width
        if ratio < 1.0:
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Encode to JPEG base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def capture_raw_jpeg(self, max_width=640, quality=40):
        """Returns raw JPEG bytes (for streaming to frontend, not LLM)."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        ratio = max_width / img.width
        if ratio < 1.0:
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


if __name__ == "__main__":
    sc = ScreenCapture()
    b64 = sc.capture()
    print(f"Screenshot captured: {len(b64)} chars base64")
