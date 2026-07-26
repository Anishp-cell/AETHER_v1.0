import sys
import os
import time
import ctypes
from ctypes import wintypes
import threading
import queue
import tkinter as tk
from tkinter import ttk

# Add parent directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Win32 Constants for Global Hotkey (Ctrl + Shift + A)
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_A = 0x41
HOTKEY_ID = 101
WM_HOTKEY = 0x0312


class DesktopHUDOverlay:
    """
    Native Windows top-most transparent floating HUD widget.
    Toggles on Ctrl + Shift + A anywhere on the computer.
    """
    def __init__(self, action_callback=None):
        self.action_callback = action_callback
        self.root = None
        self.is_visible = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.log_text = None
        self.input_entry = None
        self.log_queue = queue.Queue()
        self._thread = None


    def start_in_thread(self):
        """Starts the HUD overlay GUI in a background daemon thread."""
        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()
        self._start_hotkey_listener()

    def _run_gui(self):
        try:
            self.root = tk.Tk()
            self.root.title("AETHER Tactical HUD")
            self.root.overrideredirect(True)
            self.root.wm_attributes("-topmost", True)
            self.root.wm_attributes("-alpha", 0.94)

            # Dark Glass Palette
            bg_color = "#0F172A"       # Deep Slate
            accent_cyan = "#00F5FF"    # Neon Cyan
            border_color = "#1E293B"   # Slate Border
            text_color = "#E2E8F0"     # Light Slate Text

            self.root.config(bg=border_color)

            # Center position on top-right of screen
            screen_w = self.root.winfo_screenwidth()
            w, h = 440, 280
            x_pos = screen_w - w - 40
            y_pos = 60
            self.root.geometry(f"{w}x{h}+{x_pos}+{y_pos}")

            # Main Container Frame
            main_frame = tk.Frame(self.root, bg=bg_color, highlightbackground=accent_cyan, highlightthickness=1.5)
            main_frame.pack(fill="both", expand=True, padx=2, pady=2)

            # --- Title Bar & Drag Area ---
            title_bar = tk.Frame(main_frame, bg="#1E293B", height=32)
            title_bar.pack(fill="x", side="top")
            title_bar.bind("<Button-1>", self._start_drag)
            title_bar.bind("<B1-Motion>", self._do_drag)

            title_label = tk.Label(
                title_bar, 
                text="⚡ AETHER v3.0 // TACTICAL HUD", 
                fg=accent_cyan, 
                bg="#1E293B", 
                font=("JetBrains Mono", 9, "bold")
            )
            title_label.pack(side="left", padx=10, pady=4)
            title_label.bind("<Button-1>", self._start_drag)
            title_label.bind("<B1-Motion>", self._do_drag)

            close_btn = tk.Button(
                title_bar, 
                text="✕", 
                fg="#94A3B8", 
                bg="#1E293B", 
                activebackground="#EF4444", 
                activeforeground="#FFFFFF",
                bd=0, 
                font=("Arial", 9, "bold"),
                command=self.hide
            )
            close_btn.pack(side="right", padx=8)

            # --- Action Buttons Bar ---
            btn_frame = tk.Frame(main_frame, bg=bg_color)
            btn_frame.pack(fill="x", padx=8, pady=6)

            btn_style = {
                "font": ("Segoe UI", 8, "bold"),
                "bg": "#1E293B",
                "fg": text_color,
                "activebackground": "#334155",
                "activeforeground": accent_cyan,
                "bd": 1,
                "relief": "flat",
                "cursor": "hand2"
            }

            btn_screen = tk.Button(btn_frame, text="📸 Screen", command=self._on_click_screen, **btn_style)
            btn_screen.pack(side="left", expand=True, fill="x", padx=2)

            btn_diag = tk.Button(btn_frame, text="📊 Diagnostics", command=self._on_click_diag, **btn_style)
            btn_diag.pack(side="left", expand=True, fill="x", padx=2)

            btn_time = tk.Button(btn_frame, text="⏰ Time", command=self._on_click_time, **btn_style)
            btn_time.pack(side="left", expand=True, fill="x", padx=2)

            # --- Log & Output Feed ---
            log_frame = tk.Frame(main_frame, bg="#020617", bd=1, relief="solid")
            log_frame.pack(fill="both", expand=True, padx=8, pady=2)

            self.log_text = tk.Text(
                log_frame, 
                bg="#020617", 
                fg="#38BDF8", 
                font=("Consolas", 8), 
                wrap="word", 
                bd=0,
                highlightthickness=0
            )
            self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
            self.append_log("System HUD Ready. Press [Ctrl + Shift + A] to toggle.")

            # --- Prompt Input Entry ---
            input_frame = tk.Frame(main_frame, bg=bg_color)
            input_frame.pack(fill="x", padx=8, pady=6)

            self.input_entry = tk.Entry(
                input_frame, 
                bg="#1E293B", 
                fg="#FFFFFF", 
                insertbackground=accent_cyan,
                font=("Segoe UI", 9), 
                bd=1, 
                relief="flat"
            )
            self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 4), ipady=3)
            self.input_entry.bind("<Return>", lambda e: self._on_send_prompt())

            send_btn = tk.Button(
                input_frame, 
                text="SEND", 
                bg="#0284C7", 
                fg="#FFFFFF", 
                font=("Segoe UI", 8, "bold"), 
                bd=0, 
                cursor="hand2",
                command=self._on_send_prompt
            )
            send_btn.pack(side="right", ipadx=8, ipady=2)

            # Start hidden, show on hotkey
            self.root.withdraw()
            self.is_visible = False
            self.root.after(50, self._poll_queue)
            self.root.mainloop()

        except Exception as e:
            print(f"[Desktop HUD Error] GUI Loop failed: {e}")

    def _poll_queue(self):
        """Polls background log queue every 50ms on main GUI thread."""
        while not self.log_queue.empty():
            try:
                text = self.log_queue.get_nowait()
                if self.log_text:
                    self.log_text.insert(tk.END, f"\n> {text}")
                    self.log_text.see(tk.END)
            except Exception:
                break
        if self.root:
            self.root.after(50, self._poll_queue)

    def show(self):
        """Shows the floating HUD overlay."""
        if self.root:
            self.root.deiconify()
            self.root.wm_attributes("-topmost", True)
            self.root.focus_force()
            self.is_visible = True
            if self.input_entry:
                self.input_entry.focus_set()

    def hide(self):
        """Hides the floating HUD overlay."""
        if self.root:
            self.root.withdraw()
            self.is_visible = False

    def toggle(self):
        """Toggles HUD visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()

    def append_log(self, text: str):
        """Appends text to the HUD log widget safely from any thread."""
        self.log_queue.put(text)


    def _start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + (event.x - self.drag_start_x)
        y = self.root.winfo_y() + (event.y - self.drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _on_click_screen(self):
        self.append_log("Triggering Screen Analysis...")
        if self.action_callback:
            threading.Thread(target=self.action_callback, args=("Analyze my screen and tell me what is on it.",), daemon=True).start()

    def _on_click_diag(self):
        self.append_log("Reading System Telemetry...")
        if self.action_callback:
            threading.Thread(target=self.action_callback, args=("Get system diagnostics",), daemon=True).start()

    def _on_click_time(self):
        self.append_log("Fetching Time...")
        if self.action_callback:
            threading.Thread(target=self.action_callback, args=("What is the current time?",), daemon=True).start()

    def _on_send_prompt(self):
        if not self.input_entry:
            return
        text = self.input_entry.get().strip()
        if text:
            self.input_entry.delete(0, tk.END)
            self.append_log(f"User: {text}")
            if self.action_callback:
                threading.Thread(target=self.action_callback, args=(text,), daemon=True).start()

    def _start_hotkey_listener(self):
        """Registers global Ctrl + Shift + A hotkey via Win32 user32.dll."""
        def _hotkey_loop():
            user32 = ctypes.windll.user32
            if not user32.RegisterHotKey(None, HOTKEY_ID, MOD_CONTROL | MOD_SHIFT, VK_A):
                print("[Desktop HUD Error] Could not register global hotkey Ctrl + Shift + A!")
                return
            
            print("[Desktop HUD] ⚡ Registered System-Wide Hotkey: [Ctrl + Shift + A]")
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    print("[Desktop HUD] Hotkey [Ctrl + Shift + A] Pressed!")
                    self.toggle()
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            user32.UnregisterHotKey(None, HOTKEY_ID)

        thread = threading.Thread(target=_hotkey_loop, daemon=True)
        thread.start()


# Global Singleton HUD Instance
DESKTOP_HUD = DesktopHUDOverlay()

if __name__ == "__main__":
    def dummy_callback(prompt):
        print(f"[HUD Action] Processing prompt: '{prompt}'")
        time.sleep(1)
        DESKTOP_HUD.append_log(f"AETHER: Processed '{prompt}' successfully!")

    print("Launching Desktop HUD Overlay test... Press Ctrl + Shift + A to toggle!")
    DESKTOP_HUD.action_callback = dummy_callback
    DESKTOP_HUD.start_in_thread()
    DESKTOP_HUD.show()

    # Keep main process alive for test
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("HUD Test exited.")
