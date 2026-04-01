import requests
import re
import json
from tools_executor import AVAILABLE_TOOLS, OLLAMA_TOOL_DEFINITIONS

class FrozenLLMEngine:
    def __init__(self, base_model="qwen2.5:1.5b", thinking_model="deepseek-r1:1.5b"):
        self.base_model = base_model
        self.thinking_model = thinking_model
        self.url = "http://127.0.0.1:11434/api/chat"
        print(f"[Logic Layer] Dual-Model LLM Engine Initialized ({base_model} & {thinking_model})")

    def generate_response(self, system_prompt: str, user_input: str, chat_history: list = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject context history
        if chat_history:
            for hp in chat_history:
                role_val = "assistant" if hp.get("role") == "aether" else "user"
                content_val = hp.get("text", "")
                if content_val:
                    messages.append({"role": role_val, "content": content_val})
        
        messages.append({"role": "user", "content": user_input})
        
        # 1. Base query: The Router Model attempts to answer or use tools
        payload = {
            "model": self.base_model,
            "messages": messages,
            "stream": False,
            "tools": OLLAMA_TOOL_DEFINITIONS,
            "options": {"temperature": 0.1} # Extremely low temp for consistent function routing
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status() 
            data = response.json()
            message = data.get("message", {})
            
            # --- TOOL EXECUTION / ROUTING LAYER --- 
            if "tool_calls" in message and message["tool_calls"]:
                tool_results_history = []
                last_result_type = None
                
                # We append the original model's tool call message
                messages.append(message)
                
                # Execute all tools the model requested in order
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"].get("arguments", {})
                    
                    print(f"\n[Action Router] Micro-Expert Triggered Tool: {tool_name}")
                    print(f"[Action Router] Arguments: {tool_args}")
                    
                    if tool_name in AVAILABLE_TOOLS:
                        # ── Content Generation Intercept ──
                        # If the LLM called open_app_and_type but passed a vague instruction
                        # like "Write a poem." instead of actual content, we generate the 
                        # content first using a separate LLM call, then pass the real text.
                        if tool_name == "open_app_and_type" and "text_to_type" in tool_args:
                            text = tool_args["text_to_type"]
                            # Detect if this is an instruction rather than actual content
                            instruction_markers = ["write ", "create ", "make ", "generate ", "compose "]
                            is_instruction = any(text.lower().startswith(m) for m in instruction_markers) and len(text) < 100
                            
                            if is_instruction:
                                print(f"\n[Content Engine] Detected vague instruction: '{text}'. Generating actual content...")
                                gen_payload = {
                                    "model": self.base_model,
                                    "messages": [
                                        {"role": "system", "content": "You are a creative writer. Generate the requested content directly with no extra commentary. Do not use markdown formatting. Output ONLY the content itself."},
                                        {"role": "user", "content": f"{user_input}"}
                                    ],
                                    "stream": False,
                                    "options": {"temperature": 0.7}
                                }
                                gen_resp = requests.post(self.url, json=gen_payload).json()
                                generated_text = gen_resp.get("message", {}).get("content", text).strip()
                                # Clean markdown artifacts
                                generated_text = re.sub(r'```\w*\n?', '', generated_text)
                                generated_text = re.sub(r'\*\*', '', generated_text)
                                tool_args["text_to_type"] = generated_text
                                print(f"[Content Engine] Generated {len(generated_text)} chars of content.")
                        
                        # Execute the physical script!
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)

                        
                        # A. Dual-Model Logic Triggered
                        if tool_result == "[DEEPSEEK_ROUTING_ACTIVATED]":
                            print(f"[Model Swift] Routing prompt to Heavy Logic Engine ({self.thinking_model})...")
                            return self._generate_thinking_response(system_prompt, user_input, chat_history)
                        
                        # B. Standard Tool Triggered
                        messages.append({"role": "tool", "content": str(tool_result)}) 
                
                # After executing ALL tools, ask Qwen to summarize the results
                final_payload = {
                    "model": self.base_model,
                    "messages": messages,
                    "stream": False
                }
                final_response = requests.post(self.url, json=final_payload).json()
                return final_response["message"]["content"].strip()

            # --- NORMAL CHAT LAYER ---
            return message.get("content", "").strip()
            
        except Exception as e:
            return f"[Engine Error] connection failed: {str(e)}"

    def _generate_thinking_response(self, system_prompt: str, user_input: str, chat_history: list = None) -> str:
        """ Safely bypasses the router to hit the DeepSeek R1 reasoning chain directly. """
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            for hp in chat_history:
                role_val = "assistant" if hp.get("role") == "aether" else "user"
                content_val = hp.get("text", "")
                if content_val:
                    messages.append({"role": role_val, "content": content_val})
                    
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": self.thinking_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.6}
        }
        
        try:
            response = requests.post(self.url, json=payload).json()
            raw_text = response.get("message", {}).get("content", "")
            
            # Scrub the structural reasoning tags out so Piper TTS doesn't try to dictate AETHER's inner thoughts
            cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            return cleaned_text
        except Exception as e:
            return f"[Thinking Engine Error] {str(e)}"
