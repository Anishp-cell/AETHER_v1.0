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
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[AetherCoder] Code saved to {filepath}")
            
            # 3. Execute Code
            print(f"[AetherCoder] Executing sequence...")
            # We use the python executable from the active virtualenv
            import sys
            python_exe = sys.executable 
            
            result = subprocess.run([python_exe, filepath], capture_output=True, text=True, timeout=30)
            
            output = ""
            if result.stdout:
                output += f"Output:\n{result.stdout.strip()}"
            if result.stderr:
                output += f"\nErrors:\n{result.stderr.strip()}"
                
            if result.returncode == 0:
                print(f"[AetherCoder] Execution successful.")
                return f"Successfully wrote and executed script intended to '{instruction}'. Terminal Output: {output[:1000]}"
            else:
                print(f"[AetherCoder] Execution crashed (Return Code {result.returncode}).")
                return f"Wrote script but it crashed during execution. Crash logs: {output[:1000]}"
                
        except subprocess.TimeoutExpired:
            return f"Timeout Error: The generated script took longer than 30 seconds to execute and was killed."
        except Exception as e:
            return f"[AetherCoder Error] {str(e)}"
