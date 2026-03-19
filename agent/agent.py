import json
import requests
import os
import sqlite3
import base64
import io
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PASTE THIS IN: agent/agent.py
# VISION ENABLED — Agent can actually SEE the browser!
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "browsergym.db")

TEXT_PROMPT = """You are a precise browser automation agent. Complete this task: {task}

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
SELECTORS TO USE
═══════════════════════════════════════════
Google search      → textarea[name='q']
Google submit      → input[name='btnK']
YouTube search     → input[name='search_query']
Wikipedia search   → #searchInput
GitHub search      → input[name='q']
Any link/tab       → a:has-text('LINK TEXT')
Any button         → button:has-text('BUTTON TEXT')
Input by label     → input[placeholder='Search...']

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
1. ALWAYS click input before typing
2. value = ONLY the search term, never full instruction
3. For tabs/buttons use: a:has-text('Tab Name')
4. If selector fails try: text='Button Text'
5. When done → wait with "Task is complete, stopping."

ACTIONS: 0=click 1=type 2=scroll 3=navigate 4=submit 5=back 6=wait

RESPOND WITH ONLY JSON:
{{"action_type":<0-6>,"action_type_name":"<n>","selector":"<css>","value":"<text>","scroll_direction":1,"reasoning":"<why>"}}"""

VISION_PROMPT = """You are a browser automation agent with VISION. You can see the screenshot of the browser.

Task: {task}

═══════════════════════════════════════════
LESSONS FROM PAST EXPERIENCE
═══════════════════════════════════════════
{few_shot_examples}

═══════════════════════════════════════════
CURRENT STATE
═══════════════════════════════════════════
URL: {url}
Page Title: {title}
Step: {step_count}/20

Previous actions:
{history}

═══════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════
Look at the screenshot carefully:
1. Find the element you need to interact with
2. Use its exact text/label for the selector
3. For buttons/tabs use: a:has-text('EXACT TEXT ON BUTTON')
4. For input fields use: input[placeholder='...'] or textarea[name='q']
5. NEVER put full instruction as value — only the search term

RULES:
- ALWAYS click input field before typing
- value must be ONLY the search term
- When task is complete → wait with "Task is complete, stopping."
- If same action fails twice → try different approach

ACTIONS: 0=click 1=type 2=scroll 3=navigate 4=submit 5=back 6=wait

RESPOND WITH ONLY THIS JSON (no explanation):
{{"action_type":<0-6>,"action_type_name":"<n>","selector":"<css selector>","value":"<text or url>","scroll_direction":1,"reasoning":"<what you see and why this action>"}}"""


class BrowserAgent:
    def __init__(self):
        self.action_history = []
        self.task = ""
        self.last_used_brain = "none"
        self._stop_requested = False
        self.groq_failed = False
        self.gemini_failed = False
        self.current_screenshot_b64 = None  # Store latest screenshot

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
    def decide_action(self, observation: dict) -> dict:
        # Store screenshot as base64
        screenshot = observation.get("screenshot")
        if screenshot is not None:
            self.current_screenshot_b64 = self._screenshot_to_b64(screenshot)

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
            return None
        except:
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
            return None
        except:
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
            return None
        except:
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