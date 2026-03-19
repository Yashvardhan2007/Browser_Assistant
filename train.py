import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.browser_env import make_env
from agent.agent import BrowserAgent
from memory.replay_buffer import ReplayBuffer

# ============================================================
# PASTE THIS IN: train.py
# Hard multi-step tasks — agent must navigate from homepage!
# ============================================================

# These tasks start at homepage and require MULTIPLE steps
TRAINING_TASKS = [
    {
        "instruction": "Search for PyTorch on Google",
        "task_type": "search",
        "url": "https://www.google.com",  # Start at homepage
        "success_condition": {"url_contains": "search?q="},
    },
    {
        "instruction": "Search for machine learning on YouTube",
        "task_type": "search",
        "url": "https://www.youtube.com",  # Start at homepage
        "success_condition": {"url_contains": "results?search_query="},
    },
    {
        "instruction": "Search for reinforcement learning on Wikipedia",
        "task_type": "search",
        "url": "https://www.wikipedia.org",  # Start at homepage
        "success_condition": {"url_contains": "wiki/"},
    },
    {
        "instruction": "Search for deep learning on Google",
        "task_type": "search",
        "url": "https://www.google.com",
        "success_condition": {"url_contains": "search?q="},
    },
    {
        "instruction": "Search for OpenAI on Google",
        "task_type": "search",
        "url": "https://www.google.com",
        "success_condition": {"url_contains": "search?q="},
    },
    {
        "instruction": "Search for neural networks on YouTube",
        "task_type": "search",
        "url": "https://www.youtube.com",
        "success_condition": {"url_contains": "results?search_query="},
    },
    {
        "instruction": "Search for Meta AI on Wikipedia",
        "task_type": "search",
        "url": "https://www.wikipedia.org",
        "success_condition": {"url_contains": "wiki/"},
    },
    {
        "instruction": "Search for Python programming on Google",
        "task_type": "search",
        "url": "https://www.google.com",
        "success_condition": {"url_contains": "search?q="},
    },
]


def run_training(num_episodes: int = 20, verbose: bool = True):
    agent = BrowserAgent()
    memory = ReplayBuffer()

    print("=" * 60)
    print("🏋️  BrowserRL Training Loop")
    print(f"   Episodes: {num_episodes}")
    print(f"   Tasks start at HOMEPAGE — multiple steps needed!")
    print("=" * 60)

    for episode in range(num_episodes):
        task = TRAINING_TASKS[episode % len(TRAINING_TASKS)]

        if verbose:
            print(f"\n📍 Episode {episode+1}/{num_episodes}")
            print(f"   Task: {task['instruction']}")
            print(f"   Start URL: {task['url']}")

        agent.set_task(task["instruction"])
        env = make_env(task)

        try:
            observation, info = env.reset()
            if verbose:
                print(f"   🌐 Loaded: {observation['url']}")

            total_reward = 0
            steps_log = []

            for step in range(20):
                action = agent.decide_action(observation)
                action_name = action.get("action_type_name", "wait")
                brain = agent.last_used_brain
                url_before = observation.get("url", "")

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                steps_log.append({
                    "step": step + 1,
                    "action": action_name,
                    "selector": action.get("selector", ""),
                    "value": action.get("value", ""),
                    "reward": reward,
                    "url_before": url_before,
                    "url_after": observation.get("url", ""),
                    "brain": brain,
                })

                if verbose:
                    print(f"   Step {step+1}: {action_name} | reward: {reward:+.2f} | {brain}")

                if terminated or truncated:
                    break

            success = info.get("success", False) or terminated

            memory.save_episode(
                task=task["instruction"],
                task_type=task["task_type"],
                total_reward=total_reward,
                steps_taken=len(steps_log),
                success=success,
                final_url=observation.get("url", ""),
                brain_used=agent.last_used_brain,
                steps_log=steps_log,
            )

            status = "✅ SUCCESS" if success else "❌ FAILED"
            if verbose:
                print(f"   {status} | Reward: {total_reward:.2f} | Steps: {len(steps_log)}")

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        finally:
            env.close()
            time.sleep(1)

    # ── Summary ────────────────────────────────────────────
    summary = memory.get_summary()
    print("\n" + "=" * 60)
    print("📊 Training Complete!")
    print(f"   Total Episodes:  {summary['total_episodes']}")
    print(f"   Success Rate:    {summary['success_rate']:.1f}%")
    print(f"   Avg Reward:      {summary['avg_reward']:.2f}")
    print(f"   Avg Steps:       {summary['avg_steps']:.1f}")
    print(f"   Best Reward:     {summary['best_reward']:.2f}")
    print("=" * 60)
    print("\n🚀 View graphs: python dashboard.py")


if __name__ == "__main__":
    run_training(num_episodes=20, verbose=True)