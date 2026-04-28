import requests
import re
import json
from tools_executor import AVAILABLE_TOOLS, OLLAMA_TOOL_DEFINITIONS

class FrozenLLMEngine:
    def __init__(self, base_model="qwen2.5-coder:1.5b", thinking_model="deepseek-r1:1.5b", orchestrator_model="Aether-Orchestrator"):
        self.base_model = base_model
        self.thinking_model = thinking_model
        self.orchestrator_model = orchestrator_model
        self.url = "http://127.0.0.1:11434/api/chat"
        print(f"[Logic Layer] Triple-Model LLM Engine Initialized ({base_model}, {thinking_model}, {orchestrator_model})")

    def generate_response(self, system_prompt: str, user_input: str, chat_history: list = None, image_b64: str = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject context history
        if chat_history:
            for hp in chat_history:
                role_val = "assistant" if hp.get("role") == "aether" else "user"
                content_val = hp.get("text", "")
                if content_val:
                    # Strip internal system markers so the LLM doesn't reproduce them
                    content_val = content_val.replace(" -- [INTERRUPTED BY USER]", "").replace("[INTERRUPTED BY USER]", "")
                    content_val = re.sub(r'\[Background System Task Completed\].*', '', content_val, flags=re.DOTALL).strip()
                    if content_val:
                        messages.append({"role": role_val, "content": content_val})
        
        # --- VISUAL PRE-PROCESSING (Agent Chaining) ---
        if image_b64:
            print(f"\n👁️ [Agent Swarm] Triggering llava-phi3 for visual pre-processing...")
            vision_payload = {
                "model": "llava-phi3",
                "messages": [{
                    "role": "user",
                    "content": "Describe exactly what you see on this screen in high detail. Be specific about applications, code, text, or errors.",
                    "images": [image_b64]
                }],
                "stream": False,
                "options": {"temperature": 0.3}
            }
            try:
                vision_resp = requests.post(self.url, json=vision_payload).json()
                vision_text = vision_resp.get("message", {}).get("content", "").strip()
                print(f"[Vision Result]: {vision_text[:150]}...")
                
                # Chain the result into Qwen's prompt!
                user_input = f"{user_input}\n\n[SYSTEM INJECTED MULTIMODAL CONTEXT]: The user has shared their screen. Visual analysis engine reports: {vision_text}"
            except Exception as e:
                print(f"[Vision Layer] Failed to process image: {e}")
        
        messages.append({"role": "user", "content": user_input})
        
        # --- BEHAVIORAL OVERRIDE (CRITICAL SILENCE & AUTONOMY) ---
        action_trigger_keywords = [
            "whatsapp", "text", "message", "send", "timer", "open", "search", "find", "scrape", "google",
            "battery", "cpu", "ram", "diagnostics", "status report", "system health", "system status",
            "linkedin", "youtube", "instagram", "facebook", "netflix", "github", "reddit", "gmail", "twitter",
            "navigate", "go to", "website",
            "script", "python", "code", "mouse", "click", "scroll", "keyboard", "shortcut", "shell",
            "spotify", "song", "music", "play", "pause", "stop", "volume", "mute", "media", "tool", "skill"
        ]
        
        # Dynamically append names of all loaded tools to trigger list
        for tool in OLLAMA_TOOL_DEFINITIONS:
            name_parts = tool["function"]["name"].split("_")
            action_trigger_keywords.extend(name_parts)
        
        # Build a list of loaded tools to explicitly inject into the prompt
        loaded_tool_names = [t["function"]["name"] for t in OLLAMA_TOOL_DEFINITIONS]
        dynamic_hint = f" AVAILABLE TOOLS: {', '.join(loaded_tool_names)}."
        
        target_model = self.base_model
        if any(k.lower() in user_input.lower() for k in action_trigger_keywords):
            target_model = self.orchestrator_model
            
            # Extract minimal context for multi-turn tool calling
            context_str = ""
            if len(messages) > 1:
                last_msg = messages[-1]
                if last_msg["role"] == "assistant":
                    context_str = f"Previous context: {last_msg['content']}\n\n"
            
            # Override messages to clear full history but keep minimal context for SFT model
            messages = [
                {
                    "role": "system", 
                    "content": "You are a tool-calling orchestrator. Output ONLY a raw JSON array of tool calls. No text, no markdown, no explanation." + dynamic_hint + " CRITICAL RULE: If the user asks to USE or RUN a skill, find its exact match in the available tools and execute it. DO NOT trigger 'teach_new_skill' unless the user uses the words 'write', 'create', 'teach' or 'learn'."
                },
                {
                    "role": "user",
                    "content": context_str + user_input
                }
            ]
        
        # --- EXECUTION LAYER (Single-Pass Sequential) ---
        secret_log = ""
        
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "tools": OLLAMA_TOOL_DEFINITIONS,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status() 
            data = response.json()
            message = data.get("message", {})
            
            raw_content = message.get("content", "").strip()
            tool_calls_detected = message.get("tool_calls", [])
            
            # 1. ATTEMPT JSON FALLBACK PARSING
            clean_content = re.sub(r'^```(?:json)?\s*', '', raw_content, flags=re.MULTILINE)
            clean_content = re.sub(r'```\s*$', '', clean_content, flags=re.MULTILINE).strip()
            fallback_tools = []
            
            if clean_content:
                json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^}]*\}', clean_content, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\[\s*\{[^{}]*"name"\s*:\s*"(\w+)"', clean_content, re.DOTALL)
                
                if json_match:
                    try:
                        start_idx = clean_content.find('[')
                        start_obj = clean_content.find('{')
                        start = min(start_idx, start_obj) if start_idx != -1 and start_obj != -1 else max(start_idx, start_obj)
                        
                        stack = []; json_str = None
                        for i in range(start, len(clean_content)):
                            char = clean_content[i]
                            if char in ['{', '[']: stack.append(char)
                            elif char in ['}', ']']:
                                if stack: stack.pop()
                                if not stack:
                                    json_str = clean_content[start:i+1]
                                    break
                                    
                        if json_str:
                            parsed = json.loads(json_str)
                            fallback_tools = parsed if isinstance(parsed, list) else [parsed]
                    except Exception as e:
                        print(f"[JSON Fallback Error] {e}")

            # 2. EXECUTE TOOLS (Sequential Chaining)
            tools_to_run = fallback_tools if fallback_tools else tool_calls_detected
            
            if tools_to_run:
                tool_results_history = []
                messages.append({'role': 'assistant', 'content': raw_content if raw_content else 'Executing tools...'})
                
                for tool_call in tools_to_run:
                    # Extract name & args
                    if "function" in tool_call:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"].get("arguments", {})
                    else:
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})

                    if tool_name not in AVAILABLE_TOOLS and "instruction" in tool_args and "description" in tool_args:
                        # Auto-correct hallucinated skill name mapped to the tool name
                        print(f"🔧 [Auto-Correct] Redirecting hallucinated tool '{tool_name}' to 'teach_new_skill'")
                        tool_args["skill_name"] = tool_name
                        tool_name = "teach_new_skill"
                        
                    # Deterministic override: If Orchestrator tries to forge a new skill but the user wants to USE an existing one
                    if tool_name == "teach_new_skill":
                        # Check if any loaded dynamic tool matches the user's prompt
                        found_existing_tool = None
                        for loaded_tool in OLLAMA_TOOL_DEFINITIONS:
                            t_name = loaded_tool["function"]["name"]
                            # Skip default tools
                            if t_name in ["search_web", "open_app_and_type", "teach_new_skill", "search_and_read_web"]:
                                continue
                            
                            # Create a fuzzy match string (e.g. "get_bitcoin_price" -> "bitcoin price")
                            fuzzy_name = t_name.replace("get_", "").replace("_", " ")
                            
                            if fuzzy_name.lower() in user_input.lower() or t_name.lower() in user_input.lower():
                                found_existing_tool = t_name
                                break
                                
                        # If the user isn't using creation words, force the existing tool!
                        creation_words = ["create", "write", "teach", "learn", "make", "build"]
                        is_creation = any(cw in user_input.lower() for cw in creation_words)
                        
                        if found_existing_tool and not is_creation:
                            print(f"🔧 [Auto-Correct] Orchestrator attempted to forge '{found_existing_tool}' but user wants to USE it. Overriding.")
                            tool_name = found_existing_tool
                            tool_args = {"query": user_input} # Pass the user input as a generic query argument

                    print(f"\n[Action Router] Triggered Tool: {tool_name}")
                    
                    if tool_name in AVAILABLE_TOOLS:
                        # WhatsApp Generate Intercept (Base Model handles content creation based on earlier tool results!)
                        if tool_name == "send_whatsapp_message" and "message" in tool_args:
                            msg = str(tool_args["message"])
                            # Detect hallucinated JSON data from Orchestrator OR explicit summary requests
                            needs_generative_fill = False
                            if "{" in msg or "[" in msg or "name\":" in msg.lower():
                                needs_generative_fill = True
                            elif len(msg) < 80 and any(m in msg.lower() for m in ["descri", "explain", "tell", "about", "summary"]):
                                needs_generative_fill = True
                                
                            if needs_generative_fill:
                                clean_history = "\n".join([f"- The tool '{t[0]}' returned: {t[1]}" for t in tool_results_history])
                                gen_payload = {
                                    "model": self.base_model,
                                    "messages": [
                                        {"role": "system", "content": "You are a professional secretary drafting a WhatsApp text message. Output ONLY the raw text message to send. No pleasantries (like 'Sure!'), no formatting, no JSON, and no code blocks. Just the message. If the tool logs do not contain the requested information (e.g., if a web scrape failed or just opened a window), do not hallucinate facts. Just write: 'I searched for it but was unable to automatically retrieve the text.'"},
                                        {"role": "user", "content": f"The user requested: '{user_input}'.\n\nRecent Tool Logs:\n{clean_history}\n\nDraft the precise WhatsApp text message now."}
                                    ],
                                    "stream": False
                                }
                                gen_resp = requests.post(self.url, json=gen_payload).json()
                                tool_args["message"] = gen_resp.get("message", {}).get("content", "Task completed.").strip()
                        
                        try:
                            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(AVAILABLE_TOOLS[tool_name], **tool_args)
                                try:
                                    tool_result = future.result(timeout=15)
                                except FuturesTimeout:
                                    tool_result = f"Tool '{tool_name}' timed out after 15 seconds. Please try again."
                                    print(f"[Timeout] Tool '{tool_name}' exceeded 15s limit.")
                            if tool_result == "[DEEPSEEK_ROUTING_ACTIVATED]":
                                return self._generate_thinking_response(system_prompt, user_input, chat_history)
                        except Exception as e:
                            tool_result = f"Error executing {tool_name}: {e}"

                        # Append intermediate data back into the message array so the next tool can read it!
                        messages.append({"role": "tool", "content": str(tool_result)})
                        tool_results_history.append((tool_name, str(tool_result)))
                
                # 3. BASE-MODEL VERBAL SUMMARY
                if tool_results_history:
                    log_str = "\n".join([f"Executed {t[0]}: {t[1][:200]}..." for t in tool_results_history])
                    secret_log += f"\n{log_str}"
                    
                    print(f"[Speaking Engine] Generating natural vocal response...")
                    # Downshift back to conversational Qwen2.5 for speech
                    final_payload = {
                        "model": self.base_model,
                        "messages": [
                            {"role": "system", "content": "Your agent pipeline just executed tasks and returned data. Inform the user of the final result or answer in 1-2 conversational sentences. If the tool returned specific data (like a price, time, or a fact), tell the user the data! Do not just say 'task complete'."},
                            {"role": "user", "content": f"The user originally asked: '{user_input}'\nTool Logs: {log_str}"}
                        ],
                        "stream": False
                    }
                    try:
                        final_resp = requests.post(self.url, json=final_payload).json()
                        answer = final_resp.get("message", {}).get("content", "Task completed.").strip()
                    except Exception:
                        answer = "Task executed successfully."
                        
                    answer += f"\n\n<SECRET_TOOL_LOG>{secret_log}\n</SECRET_TOOL_LOG>"
                    return answer

            # If no tools were detected, return the raw chat text from the base model
            return raw_content

        except Exception as e:
            return f"[Engine Error] Agent failed: {str(e)}"

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
