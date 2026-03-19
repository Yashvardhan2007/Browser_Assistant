import json
import requests

# ============================================================
# PASTE THIS IN: agent/agent.py
#
# Smart fallback system:
# - Once Groq fails → skip it for whole task
# - Once Gemini fails → skip it for whole task  
# - Falls back to Ollama and stays there
# ============================================================

# ── SET YOUR API KEYS HERE ─────────────────────────────────
GROQ_API_KEY = "your_groq_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
# ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a browser automation agent. Complete this task: {task}

CURRENT STATE:
- URL: {url}
- Page Title: {title}
- Page Text: {dom_text}
- Steps done: {step_count}

Previous actions:
{history}

Available actions:
0=click, 1=type, 2=scroll, 3=navigate, 4=submit, 5=back, 6=wait

EXACT SELECTORS TO USE:
- Google search bar: textarea[name='q']
- Google submit button: input[name='btnK']
- YouTube search bar: input[name='search_query']
- Wikipedia search bar: #searchInput

HOW TO SEARCH ON GOOGLE (follow exactly):
Step 1: click textarea[name='q']
Step 2: type your query in textarea[name='q']
Step 3: click input[name='btnK']

HOW TO SEARCH ON YOUTUBE (follow exactly):
Step 1: click input[name='search_query']
Step 2: type your query in input[name='search_query']
Step 3: submit with Enter key using action_type 4

HOW TO SEARCH ON WIKIPEDIA (follow exactly):
Step 1: click #searchInput
Step 2: type your query in #searchInput
Step 3: submit with action_type 4

Reply with ONLY this JSON:
{{"action_type": <0-6>, "action_type_name": "<name>", "selector": "<selector>", "value": "<text or url>", "scroll_direction": 1, "reasoning": "<why>"}}

CRITICAL RULES:
1. ALWAYS click input field BEFORE typing
2. Use EXACT selectors listed above
3. Once task is complete return action_type 6 with reasoning "Task is complete, stopping."
4. Do NOT repeat same action twice in a row

ONLY the JSON object, no extra text."""


class BrowserAgent:
    def __init__(self):
        self.action_history = []
        self.task = ""
        self.last_used_brain = "none"
        self._stop_requested = False
        # These stay False until that API fails — then skip for whole task
        self.groq_failed = False
        self.gemini_failed = False

    def set_task(self, task: str):
        """Reset everything for a new task."""
        self.task = task
        self.action_history = []
        self._stop_requested = False
        # Reset failures for each new task — give APIs another chance
        self.groq_failed = False
        self.gemini_failed = False

    def stop(self):
        """Call this to stop the agent mid-task."""
        self._stop_requested = True

    def should_stop(self):
        return self._stop_requested

    def decide_action(self, observation: dict) -> dict:
        prompt = self._build_prompt(observation)
        action_text = None

        # ── Step 1: Try Groq (fastest) ─────────────────────
        # Skip if already failed earlier in this task
        if not self.groq_failed and GROQ_API_KEY != "your_groq_api_key_here":
            action_text = self._ask_groq(prompt)
            if action_text:
                self.last_used_brain = "⚡ Groq"
            else:
                # Failed — skip Groq for ALL remaining steps this task
                self.groq_failed = True

        # ── Step 2: Try Gemini (fast) ──────────────────────
        # Skip if already failed earlier in this task
        if not action_text and not self.gemini_failed and GEMINI_API_KEY != "your_gemini_api_key_here":
            action_text = self._ask_gemini(prompt)
            if action_text:
                self.last_used_brain = "🌟 Gemini"
            else:
                # Failed — skip Gemini for ALL remaining steps this task
                self.gemini_failed = True

        # ── Step 3: Ollama fallback (always works) ─────────
        if not action_text:
            action_text = self._ask_ollama(prompt)
            if action_text:
                self.last_used_brain = "🦙 Ollama"

        # ── All failed ─────────────────────────────────────
        if not action_text:
            self.last_used_brain = "❌ All failed"
            return {
                "action_type": 6,
                "action_type_name": "wait",
                "selector": "", "value": "",
                "scroll_direction": 1,
                "reasoning": "All AI brains failed, waiting...",
            }

        action = self._parse_action(action_text)
        self.action_history.append({
            "step": len(self.action_history) + 1,
            "action": action,
            "url": observation.get("url", ""),
            "brain": self.last_used_brain,
        })
        return action

    # ── GROQ ───────────────────────────────────────────────
    def _ask_groq(self, prompt: str):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1,
                },
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            # 429 = rate limit, anything else = error — both mean skip
            return None
        except:
            return None

    # ── GEMINI ─────────────────────────────────────────────
    def _ask_gemini(self, prompt: str):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 300, "temperature": 0.1},
                },
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return None
        except:
            return None

    # ── OLLAMA ─────────────────────────────────────────────
    def _ask_ollama(self, prompt: str):
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen2.5:0.5b",  # smallest = fastest
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except:
            return None

    # ── PROMPT BUILDER ─────────────────────────────────────
    def _build_prompt(self, observation: dict) -> str:
        history_text = "None yet"
        if self.action_history:
            history_text = ""
            for h in self.action_history[-3:]:
                a = h["action"]
                history_text += f"  Step {h['step']}: {a.get('action_type_name')} selector='{a.get('selector','')}' value='{a.get('value','')}'\n"

        return PROMPT_TEMPLATE.format(
            task=self.task,
            url=observation.get("url", ""),
            title=observation.get("page_title", ""),
            dom_text=observation.get("dom_text", "")[:500],
            step_count=observation.get("step_count", 0),
            history=history_text,
        )

    # ── ACTION PARSER ──────────────────────────────────────
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
            action_data = json.loads(action_text)
            return {
                "action_type": int(action_data.get("action_type", 6)),
                "action_type_name": action_data.get("action_type_name", "wait"),
                "selector": action_data.get("selector", ""),
                "value": action_data.get("value", ""),
                "scroll_direction": int(action_data.get("scroll_direction", 1)),
                "reasoning": action_data.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "action_type": 6,
                "action_type_name": "wait",
                "selector": "", "value": "",
                "scroll_direction": 1,
                "reasoning": f"Parse error: {e}",
            }