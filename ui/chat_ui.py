import gradio as gr
import sys, os, time
from PIL import Image
import numpy as np

# ============================================================
# PASTE THIS IN: ui/chat_ui.py
# ============================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.browser_env import make_env
from agent.agent import BrowserAgent
from tasks.task_config import parse_user_instruction
from memory.replay_buffer import ReplayBuffer

agent = BrowserAgent()
memory = ReplayBuffer()
current_env = None
current_screenshot = None
is_running = False


def observation_to_image(observation):
    try:
        screenshot = observation.get("screenshot")
        if screenshot is not None and isinstance(screenshot, np.ndarray):
            return Image.fromarray(screenshot.astype(np.uint8))
    except:
        pass
    return None


def run_browser_task(message, history):
    global current_env, current_screenshot, agent, is_running

    if not message.strip():
        yield history, None, "Please enter a task!", gr.update(value="▶ Run", variant="primary", interactive=True)
        return

    if is_running:
        yield history, current_screenshot, "⚠️ Already running! Click Stop first.", gr.update()
        return

    is_running = True
    agent._stop_requested = False

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🤔 Thinking..."}
    ]
    yield history, current_screenshot, "🤔 Understanding your task...", gr.update(value="⏹ Stop", variant="stop", interactive=True)
    time.sleep(0.5)

    # Check past experience
    summary = memory.get_summary()
    past_info = ""
    if summary["total_episodes"] > 0:
        past_info = f"\n_(🧠 Agent has {summary['total_episodes']} past episodes to learn from — {summary['success_rate']:.0f}% success rate)_"

    try:
        task_config = parse_user_instruction(message)
        agent.set_task(message)

        history[-1]["content"] = f"🌐 Launching browser...{past_info}"
        yield history, current_screenshot, f"🌐 Launching browser...", gr.update()
        time.sleep(0.5)

        if current_env:
            current_env.close()

        current_env = make_env(task_config)
        observation, info = current_env.reset()
        current_screenshot = observation_to_image(observation)

        history[-1]["content"] = f"📄 Opened: **{observation['url']}**{past_info}"
        yield history, current_screenshot, f"📄 Opened: {observation['url']}", gr.update()
        time.sleep(0.5)

        total_reward = 0
        steps_log = []

        for step in range(20):
            # Check stop
            if agent.should_stop():
                final = build_response(message, steps_log, total_reward, False, observation, stopped=True)
                history[-1]["content"] = final
                # Save to memory even if stopped
                _save_to_memory(message, task_config, steps_log, total_reward, False, observation)
                yield history, current_screenshot, "⏹ Stopped!", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

            history[-1]["content"] = f"🤖 Step {step+1}/20: AI thinking... _{agent.last_used_brain}_"
            yield history, current_screenshot, f"🤖 Step {step+1}: Thinking...", gr.update()

            action = agent.decide_action(observation)
            action_name = action.get("action_type_name", "unknown")
            reasoning = action.get("reasoning", "")
            selector = action.get("selector", "")
            value = action.get("value", "")
            brain = agent.last_used_brain

            emoji = {
                "click": "🖱️", "type": "⌨️", "scroll": "📜",
                "navigate": "🌐", "submit": "✅", "back": "⬅️", "wait": "⏳"
            }.get(action_name, "🔧")

            status = f"{emoji} **Step {step+1}: {action_name.upper()}** _{brain}_"
            if selector: status += f"\n- Target: `{selector}`"
            if value: status += f"\n- Value: `{value}`"
            status += f"\n- 💭 {reasoning}"

            history[-1]["content"] = status
            yield history, current_screenshot, status, gr.update()
            time.sleep(0.3)

            url_before = observation.get("url", "")
            observation, reward, terminated, truncated, info = current_env.step(action)
            total_reward += reward
            current_screenshot = observation_to_image(observation)

            steps_log.append({
                "step": step + 1,
                "action": action_name,
                "selector": selector,
                "value": value,
                "reward": reward,
                "url_before": url_before,
                "url_after": observation.get("url", ""),
                "brain": brain,
            })

            history[-1]["content"] = status + f"\n- 💰 Reward: `{reward:+.1f}` | Total: `{total_reward:.1f}`"
            yield history, current_screenshot, f"💰 Reward: {reward:+.1f}", gr.update()
            time.sleep(0.5)

            if terminated:
                final = build_response(message, steps_log, total_reward, True, observation)
                history[-1]["content"] = final
                # ✅ Save successful episode to memory!
                _save_to_memory(message, task_config, steps_log, total_reward, True, observation)
                yield history, current_screenshot, f"🎉 Task completed! Reward: **{total_reward:.1f}**\n💾 Saved to memory!", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

            if truncated:
                final = build_response(message, steps_log, total_reward, False, observation)
                history[-1]["content"] = final
                # Save failed episode too — learn from failures!
                _save_to_memory(message, task_config, steps_log, total_reward, False, observation)
                yield history, current_screenshot, f"⏰ Max steps. Reward: **{total_reward:.1f}**\n💾 Saved to memory!", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

        final = build_response(message, steps_log, total_reward, False, observation)
        history[-1]["content"] = final
        _save_to_memory(message, task_config, steps_log, total_reward, False, observation)
        yield history, current_screenshot, f"✅ Done! Reward: **{total_reward:.1f}**", gr.update(value="▶ Run", variant="primary", interactive=True)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, current_screenshot, error_msg, gr.update(value="▶ Run", variant="primary", interactive=True)

    is_running = False


def _save_to_memory(task, task_config, steps_log, total_reward, success, observation):
    """Save episode to SQLite database for future learning."""
    try:
        memory.save_episode(
            task=task,
            task_type=task_config.get("task_type", "unknown"),
            total_reward=total_reward,
            steps_taken=len(steps_log),
            success=success,
            final_url=observation.get("url", ""),
            brain_used="mixed",
            steps_log=steps_log,
        )
    except Exception as e:
        print(f"Memory save error: {e}")


def build_response(task, steps_log, total_reward, success, observation, stopped=False):
    if stopped:
        status = "⏹ Stopped"
    elif success:
        status = "✅ Task Completed!"
    else:
        status = "⚠️ Task Attempted"

    response = f"### {status}\n\n"
    response += f"**Task:** {task}\n\n"
    response += f"**Final URL:** {observation.get('url', 'unknown')}\n\n"
    response += f"**Steps:** {len(steps_log)} | **Reward:** {total_reward:.1f}\n\n"
    response += "**Action Log:**\n"
    for s in steps_log:
        emoji = {
            "click": "🖱️", "type": "⌨️", "scroll": "📜",
            "navigate": "🌐", "submit": "✅", "back": "⬅️", "wait": "⏳"
        }.get(s["action"], "🔧")
        response += f"{emoji} Step {s['step']}: {s['action'].upper()} `{s.get('reward', 0):+.1f}` _{s.get('brain','')}_\n"
    return response


def clear_chat():
    global current_env, current_screenshot, is_running
    if current_env:
        current_env.close()
        current_env = None
    current_screenshot = None
    is_running = False
    agent._stop_requested = False
    return [], None, "Chat cleared!", gr.update(value="▶ Run", variant="primary", interactive=True)


EXAMPLES = [
    "Search for PyTorch on Google",
    "Go to github.com",
    "Search for reinforcement learning on Wikipedia",
    "Go to huggingface.co",
    "Search for machine learning on YouTube",
]


def create_ui():
    # Get stats for header
    summary = memory.get_summary()

    with gr.Blocks(title="🤖 Browser AI Agent") as demo:
        gr.HTML(f"""
        <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:14px;margin-bottom:20px;color:white;">
            <h1 style="margin:0;font-size:2.2em;font-weight:800;">🤖 Browser AI Agent</h1>
            <p style="margin:6px 0 0 0;opacity:0.85;">OpenEnv + BrowserGym + Few-Shot Learning</p>
            <p style="margin:4px 0 0 0;font-size:0.85em;opacity:0.7;">
                🧠 {summary['total_episodes']} episodes learned | ✅ {summary['success_rate']:.0f}% success rate
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="💬 Chat", height=480)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Tell the agent what to do...",
                        label="Your instruction", scale=4, lines=1
                    )
                    run_btn = gr.Button("▶ Run", variant="primary", scale=1)
                gr.Examples(examples=EXAMPLES, inputs=msg_input, label="📝 Examples")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            with gr.Column(scale=2):
                screenshot_display = gr.Image(label="🌐 Live Browser View")
                status_display = gr.Markdown("👋 Enter a task to start!")
                gr.HTML("""
                <div style="background:#1e293b;border-radius:10px;padding:14px;margin-top:10px;color:#e2e8f0;">
                    <h4 style="margin:0 0 10px 0;color:#93c5fd;">🧠 How agent learns:</h4>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">1. 👁️ Sees browser state</p>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">2. 📚 Reads past successful episodes</p>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">3. 🤔 Decides best action</p>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">4. 💰 Gets reward</p>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">5. 💾 Saves to memory</p>
                    <p style="margin:5px 0;font-size:0.82em;color:#cbd5e1;">6. 🔁 Next time does better!</p>
                </div>
                """)

        run_btn.click(
            fn=run_browser_task,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, screenshot_display, status_display, run_btn]
        )
        msg_input.submit(
            fn=run_browser_task,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, screenshot_display, status_display, run_btn]
        )
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, screenshot_display, status_display, run_btn]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)