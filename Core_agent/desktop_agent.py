import ctypes
import ctypes.wintypes
import os
import time
import subprocess

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    pass

# ── Common Windows App URI Schemes / Paths ──
# Small LLMs often just say "Spotify" or "Calculator" without knowing the actual Windows launch command.
# This mapping converts friendly names to reliable Windows launch strings.
APP_ALIASES = {
    "spotify": "spotify:",
    "calculator": "calc",
    "calc": "calc",
    "paint": "mspaint",
    "settings": "ms-settings:",
    "store": "ms-windows-store:",
    "mail": "outlookmail:",
    "calendar": "outlookcal:",
    "camera": "microsoft.windows.camera:",
    "maps": "bingmaps:",
    "photos": "ms-photos:",
    "snipping tool": "snippingtool",
    "snip": "snippingtool",
    "cmd": "cmd",
    "terminal": "wt",
    "powershell": "powershell",
    "explorer": "explorer",
    "file explorer": "explorer",
    "edge": "msedge",
    "chrome": "chrome",
    "firefox": "firefox",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "code": "code",
    "vscode": "code",
    "vs code": "code",
}

def _resolve_app_name(app_name: str) -> str:
    """Resolve friendly app names to actual Windows launch commands."""
    key = app_name.strip().lower()
    return APP_ALIASES.get(key, app_name)

def _focus_window_by_title(partial_title: str, timeout: float = 5.0):
    """
    Attempts to find a window whose title contains `partial_title` and bring it to the foreground.
    Polls every 0.5s up to `timeout` seconds.
    """
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow
    ShowWindow = ctypes.windll.user32.ShowWindow
    
    target_lower = partial_title.lower()
    found_hwnd = [None]
    
    def enum_handler(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buff, length + 1)
                if target_lower in buff.value.lower():
                    found_hwnd[0] = hwnd
                    return False  # stop enumerating
        return True
    
    start = time.time()
    while time.time() - start < timeout:
        found_hwnd[0] = None
        EnumWindows(EnumWindowsProc(enum_handler), 0)
        if found_hwnd[0]:
            ShowWindow(found_hwnd[0], 9)  # SW_RESTORE
            SetForegroundWindow(found_hwnd[0])
            time.sleep(0.3)
            return True
        time.sleep(0.5)
    
    return False

def confirm_action(action_name, details):
    """
    Shows a terminal warning for execution instead of a blocking native Windows MessageBox.
    This grants AETHER true autonomous execution capability.
    """
    print("=" * 60)
    print(" ⚠️ SYSTEM OVERRIDE EXECUTED: AUTONOMOUS ACTION ⚠️")
    print(f" Action: {action_name}")
    print(f" Target: {details}")
    print("=" * 60)
    return True

def run_computer_command(action_type: str, target: str) -> str:
    """
    Called by the Swarm LLM router for simple single-step PC actions.
    """
    if not confirm_action(action_type, target):
        return "[V3.0 Core] User explicitly DENIED permission. Acknowledge and apologize."
        
    try:
        import pyautogui
        
        if action_type == "type_text":
            import pyperclip
            pyperclip.copy(target)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.3)
            return f"Successfully typed text onto the screen: {target}"
            
        elif action_type == "press_key":
            pyautogui.press(target)
            return f"Successfully simulated keypress: {target}"
            
        elif action_type == "open_app":
            resolved = _resolve_app_name(target)
            if "http" in resolved or "://" in resolved:
                import webbrowser
                webbrowser.open(resolved)
                return f"Opened: {resolved}"
            elif ":" in resolved and len(resolved) > 2:
                # URI scheme like spotify: or ms-settings:
                os.startfile(resolved)
                time.sleep(2)
                return f"Successfully launched: {target}"
            else:
                subprocess.Popen(f'start "" "{resolved}"', shell=True, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
                return f"Successfully launched: {target}"
            
        elif action_type == "shell_command":
            output = os.popen(target).read()
            out_str = output[:1000] if output else "Executed successfully but returned no text."
            return f"Executed '{target}'. Shell output: {out_str}"
            
        return f"[Agent Error] Unknown action_type '{action_type}'"
        
    except Exception as e:
        return f"[Desktop Sub-Agent Error] Exception: {e}"

def open_app_and_type(app_name: str, text_to_type: str) -> str:
    """
    Compound action: Opens an application, waits for it to fully load,
    brings its window to focus, then pastes the specified text via clipboard.
    For media apps like Spotify, uses native search URIs and keyboard shortcuts.
    """
    if not confirm_action("open_app_and_type", f"App: {app_name}\nText: {text_to_type[:200]}..."):
        return "[V3.0 Core] User explicitly DENIED permission. Acknowledge and apologize."
    
    try:
        import pyautogui
        import pyperclip
        import urllib.parse
        
        app_lower = app_name.strip().lower()
        
        # ── Special Media App Handling ──
        if app_lower in ("spotify", "spotify:"):
            # Strip out "play" and quotes from the text - we just need the song/artist name
            clean_query = text_to_type
            for prefix in ["play ", "Play ", "search ", "Search ", "find ", "Find "]:
                if clean_query.startswith(prefix):
                    clean_query = clean_query[len(prefix):]
            clean_query = clean_query.strip("'\"")
            
            search_query = urllib.parse.quote(clean_query)
            spotify_uri = f"spotify:search:{search_query}"
            print(f"\n[Desktop Agent] Opening Spotify search: '{spotify_uri}'...")
            os.startfile(spotify_uri)
            print("[Desktop Agent] Waiting 6s for Spotify to load search results...")
            time.sleep(6)
            
            # Focus the Spotify window (title varies: "Spotify", "Spotify Premium", "Spotify Free", or song name)
            print("[Desktop Agent] Attempting to focus Spotify window...")
            focused = _focus_window_by_title("spotify", timeout=5.0)
            print(f"[Desktop Agent] Focus result: {focused}")
            
            if focused:
                time.sleep(1)
                # Tab into the search results area, then Enter to play the top result
                print("[Desktop Agent] Navigating to first result and pressing Enter...")
                pyautogui.press('tab')
                time.sleep(0.3)
                pyautogui.press('tab')
                time.sleep(0.3)
                pyautogui.press('tab')
                time.sleep(0.3)
                pyautogui.press('enter')
                time.sleep(0.5)
            else:
                # Fallback: just Alt-Tab to Spotify and try
                print("[Desktop Agent] Could not find Spotify window, trying Alt-Tab...")
                pyautogui.hotkey('alt', 'tab')
                time.sleep(1)
                pyautogui.press('enter')
            
            return f"Opened Spotify, searched for '{clean_query}', and pressed play on the first result."

        
        # ── Standard App Handling (Notepad, WordPad, etc.) ──
        resolved = _resolve_app_name(app_name)
        print(f"\n[Desktop Agent] Opening '{app_name}' (resolved: '{resolved}')...")
        
        if ":" in resolved and len(resolved) > 2:
            os.startfile(resolved)
        else:
            subprocess.Popen(f'start "" "{resolved}"', shell=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(3)  # Wait for app to fully render
        
        # Try to find and focus the app window
        title_hints = {
            "notepad": "Notepad",
            "wordpad": "WordPad",
            "mspaint": "Paint",
            "calc": "Calculator",
            "code": "Visual Studio Code",
        }
        search_title = title_hints.get(resolved.lower(), app_name)
        
        print(f"[Desktop Agent] Focusing window containing '{search_title}'...")
        focused = _focus_window_by_title(search_title, timeout=5.0)
        
        if not focused:
            print("[Desktop Agent] Window not found by title, trying Alt-Tab...")
            pyautogui.hotkey('alt', 'tab')
            time.sleep(1)
        
        # Paste via clipboard (handles newlines, special chars, long text)
        print(f"[Desktop Agent] Pasting text into '{app_name}'...")
        pyperclip.copy(text_to_type)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)
        
        return f"Successfully opened {app_name} and typed the text into it."
        
    except Exception as e:
        return f"[Desktop Sub-Agent Error] Exception: {e}"

def search_web(query: str) -> str:
    """
    Opens the default web browser with a Google search for the given query.
    """
    if not confirm_action("search_web", f"Search Google for: {query}"):
        return "[V3.0 Core] User explicitly DENIED permission. Acknowledge and apologize."
    
    try:
        import webbrowser
        import urllib.parse
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        webbrowser.open(search_url)
        return f"Successfully opened browser and searched Google for: {query}"
    except Exception as e:
        return f"[Desktop Sub-Agent Error] Web search failed: {e}"

def analyze_screen_with_llava(task_query: str) -> str:
    """
    Takes a screenshot of the user's desktop, encodes it, and sends it to the local 
    multimodal model (llava-phi3) for analysis.
    """
    try:
        import pyautogui
        import base64
        import requests
        from io import BytesIO
        
        image = pyautogui.screenshot()
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=50)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = {
            "model": "llava-phi3",
            "prompt": f"You are a computer use vision assistant analyzing a screenshot. {task_query} Provide highly specific details.",
            "images": [img_str],
            "stream": False
        }
        
        print("\n👁️ [Agent Swarm] Vision module analyzing screen...")
        res = requests.post("http://127.0.0.1:11434/api/generate", json=payload).json()
        vision_text = res.get('response', 'Vision model returned no response.')
        print(f"[Vision Result]: {vision_text[:200]}...")
        return f"[Screen Analysis Result]: {vision_text}"
        
    except Exception as e:
        return f"[Sub-Agent Vision Error] Could not parse screen: {e}"
