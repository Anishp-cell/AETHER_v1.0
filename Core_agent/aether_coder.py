import os
import re
import requests
import subprocess
import time

def confirm_action(action_name, details):
    """Stub. Overwritten by server.py to use WebSocket auth."""
    return True

class AetherCoder:
    """
    Sub-Agent specialized in autonomously writing, saving, and executing Python scripts.
    It uses the DeepSeek-R1 logic model for high-quality code generation.
    """
    def __init__(self, workspace_dir="aether_workspace", model="qwen2.5-coder:3b"):
        self.workspace = os.path.join(os.path.dirname(os.path.dirname(__file__)), workspace_dir)
        self.model = model
        self.url = "http://127.0.0.1:11434/api/chat"
        
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        print(f"[AetherCoder] Online. Workspace bound to: {self.workspace}")

    def write_and_run_script(self, instruction: str) -> str:
        """
        Generates Python code from the instruction, saves it to a file, executes it, 
        and returns the terminal output.
        """
        if not confirm_action("AetherCoder Generation", instruction):
            return "[V3.0 Core] User explicitly DENIED permission for script generation."
            
        print(f"\n[AetherCoder] Received coding task: '{instruction}'")
        
        # 1. Generate Code using DeepSeek
        system_prompt = """You are AetherCoder, an elite autonomous AI software engineer.
You ONLY output raw, perfectly functional Python code.
You never output markdown ticks (```python) or conversational text.
Your goal is to write a self-contained python script that fulfills the user's objective. 
If libraries might be missing, use standard libraries where possible.
DO NOT wrap the code in ```python. Just output the raw code."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        try:
            print(f"[AetherCoder] Thinking engine '{self.model}' is writing code...")
            response = requests.post(self.url, json=payload).json()
            raw_text = response.get("message", {}).get("content", "")
            
            # Clean DeepSeek `<think>` reasoning tags, keep only the code
            code = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            
            # Aggressive extraction: Pull out ONLY the code block and ignore conversational hallucinated junk
            match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                code = code.replace("```python", "").replace("```", "").strip()
            
            if not code:
                return "[AetherCoder Error] The model generated empty code."
                
            # 2. Save Code to Workspace
            filename = f"script_{int(time.time())}.py"
            filepath = os.path.join(self.workspace, filename)
            
            # Record workspace files before execution to detect newly generated artifacts
            before_files = set(os.listdir(self.workspace))
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[AetherCoder] Code saved to {filepath}")
            
            # 3. Execute Code
            print(f"[AetherCoder] Executing sequence...")
            import sys
            import base64
            import csv
            python_exe = sys.executable 
            
            # Execute script inside workspace directory so relative saves land in workspace
            result = subprocess.run([python_exe, filepath], capture_output=True, text=True, timeout=30, cwd=self.workspace)
            
            stdout_text = result.stdout.strip() if result.stdout else ""
            stderr_text = result.stderr.strip() if result.stderr else ""
            output = ""
            if stdout_text:
                output += f"Output:\n{stdout_text}"
            if stderr_text:
                output += f"\nErrors:\n{stderr_text}"
                
            # Scan for newly created or modified files in workspace
            after_files = set(os.listdir(self.workspace))
            new_files = list(after_files - before_files)
            
            artifacts = []
            for nf in new_files:
                if nf == filename:
                    continue
                nf_path = os.path.join(self.workspace, nf)
                if not os.path.isfile(nf_path):
                    continue
                ext = os.path.splitext(nf)[1].lower()
                
                # Renderable Image Artifacts (Matplotlib plots, charts, graphs)
                if ext in ['.png', '.jpg', '.jpeg', '.svg']:
                    try:
                        with open(nf_path, "rb") as img_f:
                            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                            mime = "image/svg+xml" if ext == ".svg" else f"image/{ext[1:]}"
                            artifacts.append({
                                "type": "image",
                                "name": nf,
                                "src": f"data:{mime};base64,{b64_data}"
                            })
                    except Exception as _e:
                        print(f"[AetherCoder] Could not encode image artifact {nf}: {_e}")
                        
                # HTML Interactive Artifacts
                elif ext in ['.html', '.htm']:
                    try:
                        with open(nf_path, "r", encoding="utf-8", errors="ignore") as html_f:
                            artifacts.append({
                                "type": "html",
                                "name": nf,
                                "content": html_f.read()
                            })
                    except Exception as _e:
                        print(f"[AetherCoder] Could not read HTML artifact {nf}: {_e}")
                        
                # CSV Data Tables
                elif ext == '.csv':
                    try:
                        with open(nf_path, "r", encoding="utf-8", errors="ignore") as csv_f:
                            reader = list(csv.reader(csv_f))
                            if reader:
                                headers = reader[0]
                                rows = reader[1:25]  # Top 25 rows
                                artifacts.append({
                                    "type": "table",
                                    "name": nf,
                                    "headers": headers,
                                    "rows": rows
                                })
                    except Exception as _e:
                        print(f"[AetherCoder] Could not parse CSV artifact {nf}: {_e}")

            # Construct structured artifact object
            artifact_obj = {
                "instruction": instruction,
                "filename": filename,
                "code": code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": result.returncode,
                "artifacts": artifacts,
                "timestamp": int(time.time())
            }
            
            # Store globally so server.py can emit WebSocket event
            global LATEST_CODING_ARTIFACT
            LATEST_CODING_ARTIFACT = artifact_obj

            if result.returncode == 0:
                print(f"[AetherCoder] Execution successful. Found {len(artifacts)} visual artifact(s).")
                art_msg = f" Rendered {len(artifacts)} visual artifact(s) on AetherCoder Canvas." if artifacts else ""
                return f"Successfully wrote and executed script intended to '{instruction}'.{art_msg} Terminal Output: {output[:800]}"
            else:
                print(f"[AetherCoder] Execution crashed (Return Code {result.returncode}).")
                return f"Wrote script but it crashed during execution. Crash logs: {output[:800]}"
                
        except subprocess.TimeoutExpired:
            return f"Timeout Error: The generated script took longer than 30 seconds to execute and was killed."
        except Exception as e:
            return f"[AetherCoder Error] {str(e)}"

# Module-level store for WebSocket broadcast
LATEST_CODING_ARTIFACT = None

