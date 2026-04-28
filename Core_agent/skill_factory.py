import os
import re
import requests
import subprocess
import time
import shutil

def confirm_action(action_name, details):
    """Stub. Overwritten by server.py to use WebSocket auth."""
    return True

class SkillFactory:
    """
    Sub-Agent dedicated to Self-Modification and Capability Expansion.
    Writes, tests, and permanently registers new Python tools.
    """
    def __init__(self, workspace_dir="aether_workspace", skills_dir="skills", model="qwen2.5-coder:3b"):
        self.workspace = os.path.join(os.path.dirname(os.path.dirname(__file__)), workspace_dir)
        self.skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), skills_dir)
        self.model = model
        self.url = "http://127.0.0.1:11434/api/chat"
        
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.skills_dir, exist_ok=True)
        print(f"[SkillFactory] Online. Ready to forge new tools.")

    def teach_new_skill(self, skill_name: str, description: str, instruction: str) -> str:
        """
        Autonomously generates a new permanent Python tool, tests it safely, 
        and registers it into the system memory.
        """
        if not confirm_action("teach_new_skill", instruction):
            return "[V3.0 Core] User explicitly DENIED permission for forging a new skill."

        print(f"\n[SkillFactory] 🔨 Forging new skill: '{skill_name}'")
        print(f"[SkillFactory] Instruction: {instruction}")

        # Ensure valid python identifier
        skill_name = re.sub(r'[^0-9a-zA-Z_]', '', skill_name.replace(' ', '_')).lower()

        system_prompt = f"""You are the AETHER Skill Factory. You write permanent, flawless Python code that expands the AI's capabilities.
Your objective is to write a single Python function named exactly `{skill_name}`.
The function should accomplish the following instruction: {instruction}

CRITICAL RULES:
1. ONLY output raw, perfectly functional Python code. No conversational text.
2. DO NOT wrap the code in ```python or ``` ticks.
3. The function MUST include a comprehensive docstring describing what it does.
4. If your function takes arguments, you MUST use Python Type Hints (e.g. def {skill_name}(query: str): ).
5. You must include any necessary standard library `import` statements INSIDE the function, or at the top of the file.
6. The function MUST return a string describing the result of its execution. DO NOT just print. Return the string.
7. Wrap your logic in a try-except block and return the error string if it fails.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write the function {skill_name} now."}
        ]

        max_retries = 2
        for attempt in range(max_retries + 1):
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1}
            }

            try:
                if attempt == 0:
                    print(f"[SkillFactory] Thinking engine '{self.model}' is writing {skill_name}.py...")
                else:
                    print(f"[SkillFactory] ⚠️ Syntax Error detected. Auto-Healing attempt {attempt}/{max_retries}...")
                    
                response = requests.post(self.url, json=payload).json()
                raw_text = response.get("message", {}).get("content", "")
                
                # Append the response to history in case we need to retry
                messages.append({"role": "assistant", "content": raw_text})
                
                # Clean DeepSeek tags
                code = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL | re.IGNORECASE)
                if match:
                    code = match.group(1).strip()
                else:
                    code = code.replace("```python", "").replace("```", "").strip()

                if not code:
                    return "[SkillFactory Error] Generated empty code."

                # Syntax Check
                import ast
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    if attempt < max_retries:
                        error_msg = f"Your code failed with a SyntaxError: {e}\nPlease fix the syntax error and rewrite the COMPLETE function."
                        messages.append({"role": "user", "content": error_msg})
                        continue
                    else:
                        return f"[SkillFactory Error] The generated code had a syntax error and auto-healing failed after {max_retries} attempts. Error: {e}"

                # Save to temporary sandbox
                temp_filename = f"temp_{skill_name}_{int(time.time())}.py"
                temp_filepath = os.path.join(self.workspace, temp_filename)
                
                with open(temp_filepath, "w", encoding="utf-8") as f:
                    f.write(code) # Write pure code

                # If it compiles, move it to the permanent skills directory
                perm_filepath = os.path.join(self.skills_dir, f"{skill_name}.py")
                shutil.copy(temp_filepath, perm_filepath)
                
                # Hot-Reload tools
                try:
                    import tools_executor
                    import importlib
                    importlib.reload(tools_executor)
                    print(f"[SkillFactory] ✅ Skill '{skill_name}' successfully forged and hot-reloaded!")
                    return f"Successfully forged and loaded the new permanent skill '{skill_name}'. It is now available for use in this and future conversations."
                except Exception as e:
                    return f"Wrote the skill, but failed to hot-reload the tools executor: {e}. A server restart may be required."

            except Exception as e:
                return f"[SkillFactory Error] {str(e)}"
