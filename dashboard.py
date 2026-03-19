import gradio as gr
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from memory.replay_buffer import ReplayBuffer

# ============================================================
# PASTE THIS IN: dashboard.py
# Live training dashboard with reward graphs
# ============================================================

memory = ReplayBuffer()


def get_reward_chart():
    """Generate reward over episodes chart data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import numpy as np

        episodes = memory.get_all_episodes()
        if not episodes:
            return None

        episode_nums = [e[0] for e in episodes]
        rewards = [e[3] for e in episodes]
        successes = [e[5] for e in episodes]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("#0f172a")

        # ── Plot 1: Reward over episodes ──────────────────
        ax1 = axes[0, 0]
        ax1.set_facecolor("#1e293b")
        ax1.plot(episode_nums, rewards, color="#3b82f6", linewidth=2, marker="o", markersize=4)

        # Moving average
        if len(rewards) >= 3:
            window = min(3, len(rewards))
            moving_avg = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(rewards[start:i+1]) / (i - start + 1))
            ax1.plot(episode_nums, moving_avg, color="#f59e0b", linewidth=2,
                    linestyle="--", label="Moving Avg")
            ax1.legend(facecolor="#1e293b", labelcolor="white")

        ax1.set_title("📈 Reward per Episode", color="white", fontsize=11, pad=10)
        ax1.set_xlabel("Episode", color="#94a3b8")
        ax1.set_ylabel("Total Reward", color="#94a3b8")
        ax1.tick_params(colors="#94a3b8")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#334155")

        # ── Plot 2: Success rate ───────────────────────────
        ax2 = axes[0, 1]
        ax2.set_facecolor("#1e293b")
        cumulative_success = []
        for i, s in enumerate(successes):
            rate = sum(successes[:i+1]) / (i+1) * 100
            cumulative_success.append(rate)
        ax2.fill_between(episode_nums, cumulative_success, alpha=0.3, color="#22c55e")
        ax2.plot(episode_nums, cumulative_success, color="#22c55e", linewidth=2)
        ax2.set_title("✅ Success Rate %", color="white", fontsize=11, pad=10)
        ax2.set_xlabel("Episode", color="#94a3b8")
        ax2.set_ylabel("Success %", color="#94a3b8")
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="#94a3b8")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#334155")

        # ── Plot 3: Steps per episode ──────────────────────
        ax3 = axes[1, 0]
        ax3.set_facecolor("#1e293b")
        steps = [e[4] for e in episodes]
        colors = ["#22c55e" if s else "#ef4444" for s in successes]
        ax3.bar(episode_nums, steps, color=colors, alpha=0.8)
        ax3.set_title("🔢 Steps per Episode", color="white", fontsize=11, pad=10)
        ax3.set_xlabel("Episode", color="#94a3b8")
        ax3.set_ylabel("Steps", color="#94a3b8")
        ax3.tick_params(colors="#94a3b8")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#334155")

        # ── Plot 4: Brain usage pie ────────────────────────
        ax4 = axes[1, 1]
        ax4.set_facecolor("#1e293b")
        brains = [e[7] for e in episodes if e[7]]
        brain_counts = {}
        for b in brains:
            brain_counts[b] = brain_counts.get(b, 0) + 1
        if brain_counts:
            labels = list(brain_counts.keys())
            sizes = list(brain_counts.values())
            pie_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
            ax4.pie(sizes, labels=labels, colors=pie_colors[:len(labels)],
                   autopct="%1.0f%%", textprops={"color": "white"})
        ax4.set_title("🧠 AI Brain Usage", color="white", fontsize=11, pad=10)

        plt.tight_layout(pad=2.0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#0f172a", bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    except Exception as e:
        print(f"Chart error: {e}")
        return None


def get_stats_text():
    """Get formatted stats text."""
    summary = memory.get_summary()
    if summary["total_episodes"] == 0:
        return "No training data yet. Run `python train.py` first!"

    return f"""## 📊 Training Summary

| Metric | Value |
|--------|-------|
| 🎯 Total Episodes | **{summary['total_episodes']}** |
| ✅ Successful | **{summary['successful_episodes']}** |
| 📈 Success Rate | **{summary['success_rate']:.1f}%** |
| 💰 Avg Reward | **{summary['avg_reward']:.2f}** |
| 🔢 Avg Steps | **{summary['avg_steps']:.1f}** |
| 🏆 Best Reward | **{summary['best_reward']:.2f}** |
| 🎯 Best Task | _{summary['best_task']}_ |
"""


def get_episodes_table():
    """Get recent episodes as formatted text."""
    episodes = memory.get_all_episodes()
    if not episodes:
        return "No episodes yet."

    text = "## 📋 Recent Episodes\n\n"
    text += "| # | Task | Reward | Steps | Success |\n"
    text += "|---|------|--------|-------|--------|\n"

    for e in episodes[-10:][::-1]:
        success = "✅" if e[5] else "❌"
        task = e[1][:40] + "..." if len(e[1]) > 40 else e[1]
        text += f"| {e[0]} | {task} | {e[3]:.1f} | {e[4]} | {success} |\n"

    return text


def refresh_dashboard():
    """Refresh all dashboard components."""
    chart = get_reward_chart()
    stats = get_stats_text()
    table = get_episodes_table()
    return chart, stats, table


def create_dashboard():
    with gr.Blocks(title="🧠 BrowserRL Training Dashboard") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#0f172a,#1e3a5f); border-radius:14px; margin-bottom:20px; color:white;">
            <h1 style="margin:0; font-size:2em; font-weight:800;">🧠 BrowserRL Training Dashboard</h1>
            <p style="margin:8px 0 0 0; opacity:0.8;">Live training metrics — Meta PyTorch OpenEnv Hackathon 2026</p>
        </div>
        """)

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="primary", scale=1)
            gr.Markdown("_Auto-updates when you click Refresh after running `python train.py`_")

        with gr.Row():
            with gr.Column(scale=2):
                chart_display = gr.Image(label="📈 Training Graphs", height=500)
            with gr.Column(scale=1):
                stats_display = gr.Markdown(get_stats_text())
                episodes_display = gr.Markdown(get_episodes_table())

        refresh_btn.click(
            fn=refresh_dashboard,
            outputs=[chart_display, stats_display, episodes_display]
        )

        # Load on startup
        demo.load(
            fn=refresh_dashboard,
            outputs=[chart_display, stats_display, episodes_display]
        )

    return demo


if __name__ == "__main__":
    # Install matplotlib if needed
    try:
        import matplotlib
    except:
        os.system("pip install matplotlib -q")

    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True, show_error=True)