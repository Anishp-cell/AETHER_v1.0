"""
AETHER v2.0 — Local Skill Plugin Registry (~/.aether/skills/)
Extensible plugin system that auto-discovers custom Python skills/tools from ~/.aether/skills/
and registers them into AVAILABLE_TOOLS and ARN_ROUTER.
"""
import os
import sys
import glob
import inspect
import importlib.util
import time

class LocalSkillRegistry:
    """
    Plugin manager that dynamically loads custom Python skills from ~/.aether/skills/
    """
    def __init__(self):
        self.user_skills_dir = os.path.expanduser("~/.aether/skills")
        self.local_skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")
        
        # Ensure directories exist
        os.makedirs(self.user_skills_dir, exist_ok=True)
        os.makedirs(self.local_skills_dir, exist_ok=True)
        
        self.registered_skills = {}

    def discover_and_load_skills(self, available_tools_dict=None, tool_defs_list=None):
        """
        Scans skill directories for .py files, loads functions, and registers them.
        """
        skill_files = glob.glob(os.path.join(self.user_skills_dir, "*.py")) + \
                      glob.glob(os.path.join(self.local_skills_dir, "*.py"))
        
        loaded_count = 0
        for s_file in skill_files:
            if os.path.basename(s_file).startswith("__"):
                continue
            try:
                mod_name = f"aether_skill_{os.path.splitext(os.path.basename(s_file))[0]}"
                spec = importlib.util.spec_from_file_location(mod_name, s_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    
                    # Discover all exported functions
                    for name, func in inspect.getmembers(mod, inspect.isfunction):
                        if not name.startswith("_") and func.__module__ == mod_name:
                            self.registered_skills[name] = func
                            loaded_count += 1
                            
                            # Register in AVAILABLE_TOOLS
                            if available_tools_dict is not None:
                                available_tools_dict[name] = func
                                
                            # Register in OLLAMA_TOOL_DEFINITIONS
                            if tool_defs_list is not None:
                                doc = inspect.getdoc(func) or f"Custom user skill '{name}'."
                                tool_def = {
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "description": doc,
                                        "parameters": {"type": "object", "properties": {}, "required": []}
                                    }
                                }
                                tool_defs_list.append(tool_def)
                                
                            safe_file = os.path.basename(s_file).encode("ascii", "ignore").decode("ascii")
                            print(f"[Skill Registry] Successfully loaded custom skill '{name}' from '{safe_file}'.")
            except Exception as e:
                safe_file = os.path.basename(s_file).encode("ascii", "ignore").decode("ascii")
                print(f"[Skill Registry] Note loading '{safe_file}': {e}")
                
        print(f"[Skill Registry] Total active skills loaded: {loaded_count} across '{self.user_skills_dir}'.")
        return self.registered_skills

# Global Singleton Instance
SKILL_REGISTRY = LocalSkillRegistry()
