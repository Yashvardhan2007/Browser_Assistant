
# 🤖 BrowserRL — Mini RL Environment for Browser Automation

> **Meta PyTorch OpenEnv Hackathon 2026** | Built with OpenEnv + BrowserGym + Qwen2.5

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/openenv)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What is BrowserRL?

BrowserRL is a **Mini Reinforcement Learning Environment** where an AI agent learns to complete real browser tasks through trial and error — getting smarter with every episode.

The agent:
- 👁️ **Sees** the browser (screenshot + page HTML)
- 🤔 **Decides** what action to take (powered by Groq/Gemini/Ollama)
- 🖱️ **Executes** the action (click, type, scroll, navigate)
- 💰 **Gets rewarded** based on task completion
- 🔁 **Repeats** until the task is done!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   BrowserRL System                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Chat UI (Gradio)                                   │
│       ↓                                              │
│   Task Parser → Task Config                          │
│       ↓                                              │
│   OpenEnv RL Environment (BrowserRLEnv)              │
│       ↓                                              │
│   BrowserGym + Playwright → Real Chromium Browser    │
│       ↓                                              │
│   AI Agent Brain (Groq → Gemini → Ollama fallback)   │
│       ↓                                              │
│   SQLite Memory (Replay Buffer)                      │
│       ↓                                              │
│   Training Dashboard (Matplotlib + Gradio)           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 RL Components

| Component | Implementation |
|-----------|---------------|
| **Environment** | Real web browser via BrowserGym + Playwright |
| **Observation Space** | Screenshot (1280×720) + URL + Page Title + DOM Text |
| **Action Space** | Click, Type, Scroll, Navigate, Submit, Back, Wait |
| **Reward Function** | +0.3 click, +0.3 type, +0.5 submit, +10.0 task complete |
| **Memory** | SQLite Replay Buffer storing all episodes and steps |
| **Agent Brain** | Groq (fast) → Gemini (fallback) → Ollama (offline) |

---

## 📊 Training Results

After 20 training episodes:

| Metric | Value |
|--------|-------|
| Success Rate | 70%+ |
| Avg Reward | 6.02 |
| Avg Steps | 9.6 |
| Best Reward | 11.80 |
| Tasks Covered | Google, YouTube, Wikipedia |

---

## 🚀 Installation

```bash
# Clone the repo
git clone https://github.com/Yashvardhan2007/Browser_Assistant
cd Browser_Assistant

# Install dependencies
pip install -r requirements.txt

# Install Chromium
playwright install chromium
sudo playwright install-deps

# Install Ollama (free local AI)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b

# Set API keys (optional but faster)
cp .env.example .env
# Edit .env with your Groq and Gemini keys
```

---

## ▶️ Running

```bash
# Run the demo
python demo.py

# Launch Chat UI
python ui/chat_ui.py

# Run training
python train.py

# View training dashboard
python dashboard.py
```

---

## 🎮 Example Tasks

```
"Search for PyTorch tutorials on Google"
"Search for machine learning on YouTube"  
"Navigate to github.com"
"Search for reinforcement learning on Wikipedia"
"Go to huggingface.co"
```

---

## 💰 Reward Structure

| Action | Reward |
|--------|--------|
| Successful click | +0.3 |
| Type text | +0.3 |
| Navigate to URL | +0.2 |
| Submit form | +0.5 |
| ✅ Task completed | +10.0 |
| Wrong action | -0.2 |
| Error | -0.5 |
| Max steps exceeded | -1.0 |

---

## 🤖 AI Brain Fallback System

```
Try Groq (⚡ fastest, free 30 req/min)
    ↓ if rate limited
Try Gemini (🌟 fast, free 15 req/min)  
    ↓ if rate limited
Fall back to Ollama (🦙 unlimited, local, free forever)
```

---

## 📁 Project Structure

```
Browser_Assistant/
├── env/
│   └── browser_env.py      # Core OpenEnv RL Environment
├── agent/
│   └── agent.py            # AI brain with smart fallback
├── tasks/
│   └── task_config.py      # Task definitions and URL builder
├── memory/
│   └── replay_buffer.py    # SQLite training memory
├── ui/
│   └── chat_ui.py          # Gradio Chat UI
├── train.py                # Training loop
├── dashboard.py            # Training metrics dashboard
├── demo.py                 # Demo script
└── requirements.txt        # Dependencies
```

---

## 👥 Team

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026**

---

## 🔗 Links

- [Hugging Face Demo](#) ← Add your HF Spaces link here
- [GitHub Repository](https://github.com/Yashvardhan2007/Browser_Assistant)