import gradio as gr
import sys, os, time
from PIL import Image
import numpy as np
import threading

# ============================================================
# PASTE THIS IN: ui/chat_ui.py
# ============================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.browser_env import make_env
from agent.agent import BrowserAgent
from tasks.task_config import parse_user_instruction

agent = BrowserAgent()
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

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "🤔 Thinking..."}]
    yield history, current_screenshot, "🤔 Understanding your task...", gr.update(value="⏹ Stop", variant="stop", interactive=True)
    time.sleep(0.5)

    try:
        task_config = parse_user_instruction(message)
        agent.set_task(message)

        history[-1]["content"] = f"🌐 Launching browser..."
        yield history, current_screenshot, f"🌐 Launching browser for **{task_config['task_type']}** task...", gr.update()
        time.sleep(0.5)

        if current_env:
            current_env.close()

        current_env = make_env(task_config)
        observation, info = current_env.reset()
        current_screenshot = observation_to_image(observation)

        history[-1]["content"] = f"📄 Opened: **{observation['url']}**\n\n_Starting task..._"
        yield history, current_screenshot, f"📄 Opened: {observation['url']}", gr.update()
        time.sleep(0.5)

        total_reward = 0
        steps_log = []

        for step in range(20):

            # ── Check if user clicked Stop ─────────────────
            if agent.should_stop():
                final = build_response(message, steps_log, total_reward, False, observation, stopped=True)
                history[-1]["content"] = final
                yield history, current_screenshot, "⏹ Stopped by user!", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

            history[-1]["content"] = f"🤖 Step {step+1}/20: AI is thinking... _(using {agent.last_used_brain if agent.last_used_brain != 'none' else 'AI'})_"
            yield history, current_screenshot, f"🤖 Step {step+1}: Thinking...", gr.update()

            action = agent.decide_action(observation)
            action_name = action.get("action_type_name", "unknown")
            reasoning = action.get("reasoning", "")
            selector = action.get("selector", "")
            value = action.get("value", "")

            emoji = {
                "click": "🖱️", "type": "⌨️", "scroll": "📜",
                "navigate": "🌐", "submit": "✅", "back": "⬅️", "wait": "⏳"
            }.get(action_name, "🔧")

            brain_badge = agent.last_used_brain
            status = f"{emoji} **Step {step+1}: {action_name.upper()}** _{brain_badge}_"
            if selector:
                status += f"\n- Target: `{selector}`"
            if value:
                status += f"\n- Value: `{value}`"
            status += f"\n- 💭 {reasoning}"

            history[-1]["content"] = status
            yield history, current_screenshot, status, gr.update()
            time.sleep(0.3)

            observation, reward, terminated, truncated, info = current_env.step(action)
            total_reward += reward
            current_screenshot = observation_to_image(observation)
            steps_log.append({"step": step+1, "action": action_name, "reward": reward, "brain": brain_badge})

            history[-1]["content"] = status + f"\n- 💰 Reward: `{reward:+.1f}`"
            yield history, current_screenshot, f"💰 Reward: {reward:+.1f} | Total: {total_reward:.1f}", gr.update()
            time.sleep(0.5)

            if terminated:
                final = build_response(message, steps_log, total_reward, True, observation)
                history[-1]["content"] = final
                yield history, current_screenshot, f"🎉 Task completed! Total Reward: **{total_reward:.1f}**", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

            if truncated:
                final = build_response(message, steps_log, total_reward, False, observation)
                history[-1]["content"] = final
                yield history, current_screenshot, f"⏰ Max steps reached. Total: **{total_reward:.1f}**", gr.update(value="▶ Run", variant="primary", interactive=True)
                is_running = False
                return

        final = build_response(message, steps_log, total_reward, False, observation)
        history[-1]["content"] = final
        yield history, current_screenshot, f"✅ Done! Total Reward: **{total_reward:.1f}**", gr.update(value="▶ Run", variant="primary", interactive=True)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, current_screenshot, error_msg, gr.update(value="▶ Run", variant="primary", interactive=True)

    is_running = False


def stop_task():
    global agent, is_running
    agent.stop()
    return gr.update(value="⏹ Stopping...", interactive=False)


def build_response(task, steps_log, total_reward, success, observation, stopped=False):
    if stopped:
        status = "⏹ Stopped by user"
    elif success:
        status = "✅ Task Completed!"
    else:
        status = "⚠️ Task Attempted"

    response = f"### {status}\n\n"
    response += f"**Task:** {task}\n\n"
    response += f"**Final URL:** {observation.get('url', 'unknown')}\n\n"
    response += f"**Steps taken:** {len(steps_log)} | **Total Reward:** {total_reward:.1f}\n\n"
    response += "**Action Log:**\n"
    for s in steps_log:
        emoji = {
            "click": "🖱️", "type": "⌨️", "scroll": "📜",
            "navigate": "🌐", "submit": "✅", "back": "⬅️", "wait": "⏳"
        }.get(s["action"], "🔧")
        response += f"{emoji} Step {s['step']}: {s['action'].upper()} | reward: `{s['reward']:+.1f}` | brain: _{s.get('brain','?')}_\n"
    return response


def clear_chat():
    global current_env, current_screenshot, is_running
    if current_env:
        current_env.close()
        current_env = None
    current_screenshot = None
    is_running = False
    agent._stop_requested = False
    return [], None, "Chat cleared! Enter a new task to start.", gr.update(value="▶ Run", variant="primary", interactive=True)


EXAMPLES = [
    "Navigate to github.com",
    "Go to huggingface.co",
    "Navigate to wikipedia.org",
    "Go to pytorch.org",
    "Search for reinforcement learning on Google",
]


def create_ui():
    with gr.Blocks(title="🤖 Browser AI Agent") as demo:

        # ── Header ─────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:24px; background:linear-gradient(135deg,#1e3a5f,#2563eb); border-radius:14px; margin-bottom:20px; color:white;">
            <h1 style="margin:0; font-size:2.2em; font-weight:800; letter-spacing:-1px;">🤖 Browser AI Agent</h1>
            <p style="margin:8px 0 0 0; opacity:0.85; font-size:1em;">OpenEnv + BrowserGym + Groq / Gemini / Ollama</p>
            <p style="margin:4px 0 0 0; font-size:0.8em; opacity:0.6;">Meta PyTorch OpenEnv Hackathon 2026</p>
        </div>
        """)

        with gr.Row():

            # ── Left: Chat ─────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="💬 Chat with Browser Agent", height=480)

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Tell the agent what to do... e.g. 'Go to github.com'",
                        label="Your instruction",
                        scale=4,
                        lines=1,
                    )
                    run_btn = gr.Button("▶ Run", variant="primary", scale=1)

                gr.Examples(examples=EXAMPLES, inputs=msg_input, label="📝 Quick Examples")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # ── Right: Browser + Status ────────────────────
            with gr.Column(scale=2):
                screenshot_display = gr.Image(label="🌐 Live Browser View")
                status_display = gr.Markdown(
                    value="👋 Enter a task and click **Run** to start!"
                )

                # ── How it works box ───────────────────────
                gr.HTML("""
                <div style="background:#1e293b; border-radius:10px; padding:14px; margin-top:10px; color:#e2e8f0;">
                    <h4 style="margin:0 0 10px 0; color:#93c5fd; font-size:0.95em;">🧠 How it works</h4>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">1. 👁️ Agent <b style="color:#93c5fd;">sees</b> the browser page</p>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">2. ⚡ <b style="color:#93c5fd;">Groq</b> decides the action (fastest)</p>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">3. 🌟 Falls back to <b style="color:#93c5fd;">Gemini</b> if needed</p>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">4. 🦙 Falls back to <b style="color:#93c5fd;">Ollama</b> if needed</p>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">5. 💰 Gets <b style="color:#93c5fd;">reward</b> for each good action</p>
                    <p style="margin:5px 0; font-size:0.82em; color:#cbd5e1;">6. 🔁 <b style="color:#93c5fd;">Repeats</b> until task is done!</p>
                </div>
                """)

        # ── Event Handlers ──────────────────────────────────
        run_btn.click(
            fn=run_browser_task,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, screenshot_display, status_display, run_btn],
        )
        msg_input.submit(
            fn=run_browser_task,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, screenshot_display, status_display, run_btn],
        )
        run_btn.click(
            fn=stop_task,
            outputs=[run_btn],
            cancels=[],
        )
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, screenshot_display, status_display, run_btn],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)