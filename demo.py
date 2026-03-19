"""
demo.py — Required submission file for Meta PyTorch OpenEnv Hackathon

This script demonstrates the BrowserRL environment working end-to-end:
1. Creates the RL environment
2. Runs a sample task
3. Shows observations, actions, and rewards
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.browser_env import make_env, get_env_info
from tasks.task_config import get_task_config

def run_demo():
    print("=" * 60)
    print("🤖 BrowserRL - Mini RL Environment Demo")
    print("   Meta PyTorch OpenEnv Hackathon 2026")
    print("=" * 60)

    # ── Show Environment Info ──────────────────────────────
    print("\n📋 Environment Info:")
    env_info = get_env_info()
    for key, value in env_info.items():
        print(f"   {key}: {value}")

    # ── Create a simple search task ────────────────────────
    print("\n🎯 Creating task: Search for 'PyTorch' on Google")
    task_config = get_task_config("search", query="PyTorch")
    print(f"   Task type: {task_config['task_type']}")
    print(f"   Start URL: {task_config['url']}")
    print(f"   Instructions: {task_config['instructions']}")

    # ── Initialize Environment ─────────────────────────────
    print("\n🌐 Initializing BrowserRL Environment...")
    env = make_env(task_config)

    # ── Reset (start episode) ──────────────────────────────
    print("\n🔄 Resetting environment (starting new episode)...")
    observation, info = env.reset()

    print(f"   ✅ Browser launched!")
    print(f"   📄 Current URL: {observation['url']}")
    print(f"   📝 Page Title: {observation['page_title']}")
    print(f"   🖼️  Screenshot shape: {observation['screenshot'].shape}")
    print(f"   📊 Step count: {observation['step_count']}")

    # ── Take sample actions ────────────────────────────────
    print("\n🎮 Taking sample actions:")

    # Action 1: Click search bar
    print("\n   Step 1: Click search bar")
    action1 = {
        "action_type": 0,  # click
        "selector": "input[name='q']",
        "value": "",
        "scroll_direction": 1,
    }
    obs, reward, terminated, truncated, info = env.step(action1)
    print(f"   Reward: {reward:+.2f} | URL: {obs['url'][:50]}")

    # Action 2: Type search query
    print("\n   Step 2: Type 'PyTorch'")
    action2 = {
        "action_type": 1,  # type
        "selector": "input[name='q']",
        "value": "PyTorch",
        "scroll_direction": 1,
    }
    obs, reward, terminated, truncated, info = env.step(action2)
    print(f"   Reward: {reward:+.2f} | URL: {obs['url'][:50]}")

    # Action 3: Submit search
    print("\n   Step 3: Submit search")
    action3 = {
        "action_type": 4,  # submit
        "selector": "input[name='q']",
        "value": "",
        "scroll_direction": 1,
    }
    obs, reward, terminated, truncated, info = env.step(action3)
    print(f"   Reward: {reward:+.2f} | URL: {obs['url'][:80]}")
    print(f"   Task completed: {info.get('task_completed', False)}")

    # ── Show reward structure ──────────────────────────────
    print("\n💰 Reward Structure:")
    print("   Click on element:      +0.3")
    print("   Type text:             +0.3")
    print("   Navigate to URL:       +0.2")
    print("   Submit form:           +0.5")
    print("   Task completed:       +10.0 🎉")
    print("   Wrong action:          -0.2")
    print("   Max steps exceeded:    -1.0")

    # ── Close environment ──────────────────────────────────
    env.close()
    print("\n✅ Demo completed successfully!")
    print("\n🚀 To run the full Chat UI:")
    print("   python ui/chat_ui.py")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()