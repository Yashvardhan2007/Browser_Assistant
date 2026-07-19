import json
import requests
import os
import sqlite3
import base64
import io
from dotenv import load_dotenv

load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "browsergym.db")

TEXT_PROMPT = """You are a precise browser automation agent. Complete this task: {task}
             Task Complete (Navigation) → If the goal is just to navigate, output: {{"action_type_name": "stop"}}
             Task Complete (Extraction) → If the goal is to find data, output: {{"action_type_name": "answer", "value": "THE EXACT EXTRACTED TEXT"}}
             
═══════════════════════════════════════════
ALLOWED ACTIONS
═══════════════════════════════════════════
    Click a button     → {{"action_type_name": "click", "selector": "button_id"}}
    Type text          → {{"action_type_name": "type", "selector": "input_id", "value": "text"}}
    Stop/Done          → {{"action_type_name": "stop"}}
    Extract Data       → {{"action_type_name": "answer", "value": "THE EXACT TEXT YOU FOUND"}}
═══════════════════════════════════════════
LESSONS FROM PAST EXPERIENCE
═══════════════════════════════════════════
{few_shot_examples}

═══════════════════════════════════════════
CURRENT BROWSER STATE
═══════════════════════════════════════════
URL: {url}
Page Title: {title}
Step: {step_count}/20
Visible Text: {dom_text}

Previous actions:
{history}

═══════════════════════════════════════════
🛑 CRITICAL TERMINATION RULES 🛑
═══════════════════════════════════════════
1. If your task instruction says "STOP immediately" or "URL contains:", and the current URL matches that target, YOUR TASK IS 100% COMPLETE.
2. DO NOT attempt to sign in, accept cookies, click pop-ups, or explore the page unless explicitly asked.
3. The moment the target page loads or the goal is met, you MUST output:
   {{"action_type": 6, "action_type_name": "stop", "selector": "", "value": "", "scroll_direction": 1, "reasoning": "Target reached."}}

RESPOND WITH ONLY JSON:
{{"action_type":<0-6>,"action_type_name":"<name>","selector":"<css>","value":"<text>","scroll_direction":1,"reasoning":"<why>"}}"""

VISION_PROMPT = """You are a browser automation agent with VISION. You can see the screenshot of the browser.
         - If the task is just navigation → wait with "Task is complete, stopping." (use "stop" action)
- If the task is data extraction → you MUST output the requested data inside the "value" field and use the "answer" action.

Task: {task}

═══════════════════════════════════════════
CURRENT STATE
═══════════════════════════════════════════
URL: {url}
Page Title: {title}
Step: {step_count}/20

Previous actions:
{history}

═══════════════════════════════════════════
🛑 CRITICAL TERMINATION RULES 🛑
═══════════════════════════════════════════
If the user asks you to extract, find, or summarize data, you MUST write the final extracted data in your response text explicitly BEFORE calling the STOP command.
Example:
Data: [Insert extracted links here]
Action: STOP
1. If your task instruction says "STOP immediately" or "URL contains:", and the current URL matches that target, YOUR TASK IS 100% COMPLETE.
2. DO NOT attempt to sign in, accept cookies, click pop-ups, or explore the page unless explicitly asked.
3. The moment the target page loads or the goal is met, you MUST output:
   {{"action_type": 6, "action_type_name": "stop", "selector": "", "value": "", "scroll_direction": 1, "reasoning": "Target reached."}}

ACTIONS: 0=click 1=type 2=scroll 3=navigate 4=submit 5=back 6=stop

RESPOND WITH ONLY THIS JSON (no explanation):
{{"action_type":<0-6>,"action_type_name":"<name>","selector":"<css>","value":"<text>","scroll_direction":1,"reasoning":"<what you see and why>"}}"""

class BrowserAgent:
    def __init__(self):
        self.action_history = []
        self.task = ""
        self.last_used_brain = "none"
        self._stop_requested = False
        self.groq_failed = False
        self.gemini_failed = False
        self.current_screenshot_b64 = None
        self.consecutive_waits = 0
        self.last_action_signature = None  # Store latest screenshot

    def set_task(self, task: str):
        self.task = task
        self.action_history = []
        self._stop_requested = False
        self.groq_failed = False
        self.gemini_failed = False
        self.current_screenshot_b64 = None

    def stop(self):
        self._stop_requested = True

    def should_stop(self):
        return self._stop_requested

    # ── Convert screenshot to base64 ───────────────────────
    def _screenshot_to_b64(self, screenshot_array) -> str:
        """Convert numpy screenshot array to base64 string."""
        try:
            from PIL import Image
            import numpy as np
            if screenshot_array is None:
                return ""
            img = Image.fromarray(screenshot_array.astype('uint8'))
            # Resize to smaller size to save tokens
            img = img.resize((800, 450))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            return ""

    # ── Few-shot learning from SQLite ──────────────────────
    def _get_few_shot_examples(self) -> str:
        try:
            if not os.path.exists(DB_PATH):
                return "No past experience yet."

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            keywords = self.task.lower().split()
            keyword = keywords[0] if keywords else ""

            c.execute("""
                SELECT e.id, e.task, e.total_reward, e.steps_taken
                FROM episodes e
                WHERE e.success = 1
                AND LOWER(e.task) LIKE ?
                ORDER BY e.total_reward DESC
                LIMIT 2
            """, (f"%{keyword}%",))
            similar = c.fetchall()

            if not similar:
                c.execute("""
                    SELECT e.id, e.task, e.total_reward, e.steps_taken
                    FROM episodes e
                    WHERE e.success = 1
                    ORDER BY e.total_reward DESC
                    LIMIT 3
                """)
                similar = c.fetchall()

            if not similar:
                conn.close()
                return "No successful episodes yet — learning from scratch!"

            examples = ""
            for ep in similar:
                ep_id, ep_task, ep_reward, ep_steps = ep
                c.execute("""
                    SELECT step_number, action_type, selector, value, reward
                    FROM steps WHERE episode_id = ?
                    AND reward > 0
                    ORDER BY step_number ASC
                """, (ep_id,))
                steps = c.fetchall()

                examples += f"\n✅ Past success (reward={ep_reward:.1f}):\n"
                examples += f"   Task: {ep_task}\n"
                for s in steps:
                    examples += f"   Step {s[0]}: {s[1].upper()} selector='{s[2]}' value='{s[3]}'\n"

            conn.close()
            return examples or "No relevant past experience."
        except Exception as e:
            return f"Memory error: {e}"

    # ── Main decision function ─────────────────────────────
    # ── Main decision function ─────────────────────────────
    def decide_action(self, observation: dict) -> dict:
        # 🛑 Deterministic Kill Switch: Bypass LLMs entirely if goal is met
        task_lower = self.task.lower()
        current_url = observation.get("url", "").lower()

        # Check multi-step task completion
        if "url contains:" in task_lower:
            target = task_lower.split("url contains:")[1].strip().strip("'\"")
            if target and target in current_url:
                print(f"✅ System Intervention: URL contains '{target}'. Forcing STOP action.")
                return {"action_type": 6, "action_type_name": "stop", "selector": "", "value": "", "scroll_direction": 1, "reasoning": "System determined target URL was reached."}
                
        # Check single-step strict navigation completion
        if "stop immediately" in task_lower and "you are at" in task_lower:
            target = task_lower.split("you are at")[1].split(". stop")[0].strip()
            clean_target = target.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
            if clean_target and clean_target in current_url:
                print(f"✅ System Intervention: Reached {clean_target}. Forcing STOP action.")
                return {"action_type": 6, "action_type_name": "stop", "selector": "", "value": "", "scroll_direction": 1, "reasoning": "System determined target URL was reached."}

        # Store screenshot as base64
        screenshot = observation.get("screenshot")
        if screenshot is not None:
            self.current_screenshot_b64 = self._screenshot_to_b64(screenshot)
            
        # ... [rest of your decide_action code remains exactly the same] ...
        action_text = None

        # 1. Try Groq with VISION first
        if not self.groq_failed and GROQ_API_KEY and self.current_screenshot_b64:
            action_text = self._ask_groq_vision(observation)
            if action_text:
                self.last_used_brain = "⚡ Groq 👁️"
            else:
                # Vision failed — try text only
                action_text = self._ask_groq_text(observation)
                if action_text:
                    self.last_used_brain = "⚡ Groq"
                else:
                    self.groq_failed = True

        # 2. Try Gemini with VISION
        if not action_text and not self.gemini_failed and GEMINI_API_KEY:
            action_text = self._ask_gemini_vision(observation)
            if action_text:
                self.last_used_brain = "🌟 Gemini 👁️"
            else:
                action_text = self._ask_gemini_text(observation)
                if action_text:
                    self.last_used_brain = "🌟 Gemini"
                else:
                    self.gemini_failed = True

        # 3. Ollama fallback (text only)
        if not action_text:
            action_text = self._ask_ollama(observation)
            if action_text:
                self.last_used_brain = "🦙 Ollama"

        if not action_text:
            self.last_used_brain = "❌ All failed"
            return {
                "action_type": 6, "action_type_name": "wait",
                "selector": "", "value": "", "scroll_direction": 1,
                "reasoning": "All AI brains failed.",
            }

        action = self._parse_action(action_text)
        action_name = action.get("action_type_name", "unknown").lower()
        selector = action.get("selector", "")
        current_signature = f"{action_name}_{selector}"
        
        # Check if the agent is outputting "wait" or repeating the exact same selector action
        if action_name == "wait" or current_signature == self.last_action_signature:
            self.consecutive_waits += 1
        else:
            self.consecutive_waits = 0  # Reset counter if it makes a genuine new move
            
        self.last_action_signature = current_signature
        
        # If the same brain stalls for 3 consecutive steps, force it out!
        if self.consecutive_waits >= 3:
            print(f"⚠️ {self.last_used_brain} got stuck in a loop/wait pattern! Forcing fallback...")
            self.consecutive_waits = 0  # Reset for the next brain
            self.last_action_signature = None
            
            # Disable the stuck brain so the waterfall skips it next loop
            if "Groq" in self.last_used_brain:
                self.groq_failed = True
            elif "Gemini" in self.last_used_brain:
                self.gemini_failed = True
                
            # Recursively re-run decide_action right now with the same observation data.
            # Because we flagged the current model as failed, it will drop to the next option!
            return self.decide_action(observation)
        # 🚨 --- END OF LOOP BREAKER --- 🚨

        self.action_history.append({
            "step": len(self.action_history) + 1,
            "action": action,
            "url": observation.get("url", ""),
            "brain": self.last_used_brain,
        })
        return action

    # ── Groq Vision ────────────────────────────────────────
    def _ask_groq_vision(self, observation: dict):
        try:
            prompt = self._build_vision_prompt(observation)
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.current_screenshot_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }],
                    "max_tokens": 300,
                    "temperature": 0.1,
                },
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return None
        except:
            return None

    # ── Groq Text only ─────────────────────────────────────
    def _ask_groq_text(self, observation: dict):
        try:
            prompt = self._build_text_prompt(observation)
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1,
                },
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return None
        except:
            return None

   # ── Gemini Vision ──────────────────────────────────────
    def _ask_gemini_vision(self, observation: dict):
        try:
            prompt = self._build_vision_prompt(observation)
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": self.current_screenshot_b64
                                }
                            },
                            {"text": prompt}
                        ]
                    }],
                    "generationConfig": {"maxOutputTokens": 300, "temperature": 0.1},
                },
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            # 🚨 THIS WILL TELL US IF GOOGLE REJECTS IT
            print(f"❌ Gemini Vision API Rejected: {r.status_code} - {r.text}")
            return None
        except Exception as e:
            # 🚨 THIS TELLS US IF PYTHON CRASHES
            print(f"❌ Gemini Vision Crash: {str(e)}")
            return None

    # ── Gemini Text only ───────────────────────────────────
    def _ask_gemini_text(self, observation: dict):
        try:
            prompt = self._build_text_prompt(observation)
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 300, "temperature": 0.1},
                },
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            print(f"❌ Gemini Text API Rejected: {r.status_code} - {r.text}")
            return None
        except Exception as e:
            print(f"❌ Gemini Text Crash: {str(e)}")
            return None

    # ── Ollama text only ───────────────────────────────────
    def _ask_ollama(self, observation: dict):
        try:
            prompt = self._build_text_prompt(observation)
            r = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
                
            print(f"❌ Ollama API Rejected: {r.status_code} - {r.text}")
            return None
        except Exception as e:
            print(f"❌ Ollama Crash: {str(e)}")
            return None

    # ── Prompt builders ────────────────────────────────────
    def _build_history(self) -> str:
        if not self.action_history:
            return "None yet"
        text = ""
        for h in self.action_history[-5:]:
            a = h["action"]
            text += f"  Step {h['step']}: {a.get('action_type_name','?').upper()} selector='{a.get('selector','')}' value='{a.get('value','')}' → {h['url']}\n"
        return text

    def _build_vision_prompt(self, observation: dict) -> str:
        return VISION_PROMPT.format(
            task=self.task,
            few_shot_examples=self._get_few_shot_examples(),
            url=observation.get("url", ""),
            title=observation.get("page_title", ""),
            step_count=observation.get("step_count", 0),
            history=self._build_history(),
        )

    def _build_text_prompt(self, observation: dict) -> str:
        return TEXT_PROMPT.format(
            task=self.task,
            few_shot_examples=self._get_few_shot_examples(),
            url=observation.get("url", ""),
            title=observation.get("page_title", ""),
            dom_text=observation.get("dom_text", "")[:800],
            step_count=observation.get("step_count", 0),
            history=self._build_history(),
        )

    # ── Action parser ──────────────────────────────────────
    def _parse_action(self, action_text: str) -> dict:
        try:
            action_text = action_text.strip()
            if "```json" in action_text:
                action_text = action_text.split("```json")[1].split("```")[0]
            elif "```" in action_text:
                action_text = action_text.split("```")[1].split("```")[0]
            start = action_text.find("{")
            end = action_text.rfind("}") + 1
            if start != -1 and end != 0:
                action_text = action_text[start:end]
            d = json.loads(action_text)
            return {
                "action_type": int(d.get("action_type", 6)),
                "action_type_name": d.get("action_type_name", "wait"),
                "selector": d.get("selector", ""),
                "value": d.get("value", ""),
                "scroll_direction": int(d.get("scroll_direction", 1)),
                "reasoning": d.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "action_type": 6, "action_type_name": "wait",
                "selector": "", "value": "", "scroll_direction": 1,
                "reasoning": f"Parse error: {e}",
            }