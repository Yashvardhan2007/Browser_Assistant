import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import time

# ============================================================
# PASTE THIS IN: env/browser_env.py
# ============================================================

class BrowserRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    ACTION_TYPES = ["click", "type", "scroll", "navigate", "submit", "back", "wait"]

    def __init__(self, task_config: dict, render_mode: Optional[str] = None):
        super().__init__()
        self.task_config = task_config
        self.render_mode = render_mode
        self.browser = None
        self.page = None
        self.step_count = 0
        self.max_steps = 20
        self.task_completed = False
        self._playwright_instance = None

        self.observation_space = spaces.Dict({
            "screenshot": spaces.Box(low=0, high=255, shape=(720, 1280, 3), dtype=np.uint8),
            "url": spaces.Text(max_length=500),
            "page_title": spaces.Text(max_length=200),
            "dom_text": spaces.Text(max_length=2000),
            "step_count": spaces.Discrete(21),
        })

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(self.ACTION_TYPES)),
            "selector": spaces.Text(max_length=200),
            "value": spaces.Text(max_length=500),
            "scroll_direction": spaces.Discrete(2),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.task_completed = False
        self._launch_browser()
        start_url = self.task_config.get("url", "https://www.google.com")
        self.page.goto(start_url)
        time.sleep(2)
        observation = self._get_observation()
        info = {"task": self.task_config.get("instructions", ""), "url": start_url}
        return observation, info

    def step(self, action: dict):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        try:
            action_type = self.ACTION_TYPES[action["action_type"]]
            selector = action.get("selector", "")
            value = action.get("value", "")

            if action_type == "click":
                reward += self._action_click(selector)
            elif action_type == "type":
                reward += self._action_type(selector, value)
            elif action_type == "scroll":
                reward += self._action_scroll(action.get("scroll_direction", 1))
            elif action_type == "navigate":
                reward += self._action_navigate(value)
            elif action_type == "submit":
                reward += self._action_submit(selector)
            elif action_type == "back":
                self.page.go_back()
                time.sleep(1)
                reward += -0.1
            elif action_type == "wait":
                time.sleep(1)
                reward += 0.0
                # If agent says wait, check if task is already done
                # Also if reasoning says complete, stop immediately
                reasoning = action.get("reasoning", "").lower()
                if any(word in reasoning for word in ["complete", "done", "finished", "stop", "task is complete"]):
                    terminated = True
                    self.task_completed = True
                    info["success"] = True
                    info["message"] = "✅ Task completed!"

            task_done, completion_reward = self._check_task_completion()
            if task_done:
                reward += completion_reward
                terminated = True
                self.task_completed = True
                info["success"] = True
                info["message"] = "✅ Task completed!"

        except Exception as e:
            reward += -0.5
            info["error"] = str(e)

        if self.step_count >= self.max_steps:
            truncated = True
            if not self.task_completed:
                reward += -1.0
                info["message"] = "❌ Max steps reached"

        observation = self._get_observation()
        info["step"] = self.step_count
        info["task_completed"] = self.task_completed
        info["current_url"] = self.page.url if self.page else ""

        return observation, reward, terminated, truncated, info

    def _action_click(self, selector):
        try:
            if selector:
                self.page.click(selector, timeout=5000)
            time.sleep(1)
            return 0.3
        except:
            return -0.2

    def _action_type(self, selector, text):
        try:
            if selector:
                self.page.fill(selector, text)
            time.sleep(0.5)
            return 0.3
        except:
            return -0.2

    def _action_scroll(self, direction):
        try:
            if direction == 0:
                self.page.keyboard.press("PageUp")
            else:
                self.page.keyboard.press("PageDown")
            time.sleep(0.5)
            return 0.1
        except:
            return -0.1

    def _action_navigate(self, url):
        try:
            if not url.startswith("http"):
                url = "https://" + url
            self.page.goto(url, timeout=10000)
            time.sleep(2)
            return 0.2
        except:
            return -0.3

    def _action_submit(self, selector):
        try:
            if selector:
                self.page.click(selector)
            else:
                self.page.keyboard.press("Enter")
            time.sleep(2)
            return 0.5
        except:
            return -0.2

    def _check_task_completion(self):
        success_condition = self.task_config.get("success_condition", {})
        try:
            current_url = self.page.url
            url_check = success_condition.get("url_contains", "")
            text_check = success_condition.get("page_contains", "")

            if url_check and url_check in current_url:
                return True, 10.0

            if text_check:
                content = self.page.content()
                if text_check.lower() in content.lower():
                    return True, 10.0
        except:
            pass
        return False, 0.0

    def _get_observation(self):
        try:
            screenshot_bytes = self.page.screenshot()
            screenshot = self._process_screenshot(screenshot_bytes)
            url = self.page.url
            title = self.page.title()
            dom_text = self.page.evaluate("""
                () => {
                    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    let text = '';
                    let node;
                    while (node = walker.nextNode()) { text += node.textContent + ' '; }
                    return text.slice(0, 2000);
                }
            """)
            return {
                "screenshot": screenshot,
                "url": url[:500],
                "page_title": title[:200],
                "dom_text": dom_text[:2000],
                "step_count": self.step_count,
            }
        except Exception as e:
            return {
                "screenshot": np.zeros((720, 1280, 3), dtype=np.uint8),
                "url": "", "page_title": "",
                "dom_text": f"Error: {str(e)}",
                "step_count": self.step_count,
            }

    def _process_screenshot(self, screenshot_bytes):
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(screenshot_bytes))
            img = img.resize((1280, 720)).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def _launch_browser(self):
        try:
            from playwright.sync_api import sync_playwright
            if self._playwright_instance is None:
                self._playwright_instance = sync_playwright().start()
            self.browser = self._playwright_instance.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            self.page = self.browser.new_page()
            self.page.set_viewport_size({"width": 1280, "height": 720})
        except Exception as e:
            raise RuntimeError(f"Failed to launch browser: {e}")

    def render(self):
        if self.render_mode == "rgb_array" and self.page:
            return self._get_observation()["screenshot"]

    def close(self):
        try:
            if self.browser:
                self.browser.close()
            if self._playwright_instance:
                self._playwright_instance.stop()
                self._playwright_instance = None
        except:
            pass


def make_env(task_config: dict) -> BrowserRLEnv:
    return BrowserRLEnv(task_config=task_config)


def get_env_info() -> dict:
    return {
        "name": "BrowserRL-v1",
        "description": "Mini RL Environment for Browser Automation",
        "version": "1.0.0",
        "task_types": ["form_fill", "search", "navigate"],
        "action_types": BrowserRLEnv.ACTION_TYPES,
        "max_steps": 20,
        "reward_range": (-10, 10),
    }