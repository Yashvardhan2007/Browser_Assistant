# 🤖 BrowserRL — Mini RL Environment for Browser Automation



[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/openenv)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What is BrowserRL?

BrowserRL is a **Mini Reinforcement Learning Environment** where an AI agent learns to complete real browser tasks through trial and error — getting smarter with every episode.

The agent:
- 👁️ **Sees** the browser (actual screenshot via Vision AI)
- 🤔 **Decides** what action to take (Groq Vision → Gemini Vision → Ollama)
- 🖱️ **Executes** the action (click, type, scroll, navigate)
- 💰 **Gets rewarded** based on task completion
- 💾 **Remembers** past successful episodes (SQLite few-shot learning)
- 🔁 **Gets smarter** with every episode!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BrowserRL System v2.0                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Chat UI (Gradio)                                       │
│       ↓                                                  │
│   Smart Task Parser (detects website + final goal)       │
│       ↓                                                  │
│   OpenEnv RL Environment (BrowserRLEnv)                  │
│       ↓                                                  │
│   BrowserGym + Playwright → Real Chromium Browser        │
│       ↓                                                  │
│   Vision AI Agent Brain:                                 │
│     ⚡ Groq Vision (llama-4-scout) — fastest             │
│     🌟 Gemini Vision — fallback                          │
│     🦙 Ollama (qwen2.5) — offline fallback               │
│       ↓                                                  │
│   Few-Shot Learning (reads past episodes from SQLite)    │
│       ↓                                                  │
│   SQLite Replay Buffer (stores all episodes + steps)     │
│       ↓                                                  │
│   Training Dashboard (Matplotlib + Gradio)               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 RL Components

| Component | Implementation |
|-----------|---------------|
| **Environment** | Real web browser via BrowserGym + Playwright |
| **Observation Space** | Screenshot (1280×720) + URL + Page Title + DOM Text |
| **Action Space** | Click, Type, Scroll, Navigate, Submit, Back, Wait |
| **Reward Function** | +0.3 click, +0.3 type, +0.5 submit, +10.0 task complete |
| **Memory** | SQLite Replay Buffer — stores every episode and step |
| **Learning** | Few-shot learning from past successful episodes |
| **Agent Brain** | Groq Vision → Gemini Vision → Ollama fallback |
| **Sessions** | Persistent browser sessions — agent stays logged in |

---

## 🆕 New Features in v2.0

### 👁️ Vision Support
Agent now sends actual screenshots to AI — can SEE buttons, tabs, and links instead of guessing from text!

### 🧠 Few-Shot Learning
Agent reads past successful episodes from SQLite database and uses them as examples — gets smarter with every task!

### 🔑 Persistent Sessions
Save your login once — agent reuses it forever. No more wasting steps on login!

### 🎯 Smart Task Parser
Detects specific websites and final goals automatically:
- Detects 20+ known websites (GitHub, PyPI, npm, Reddit etc.)
- Finds final goal from instruction (Issues tab, Release history etc.)
- Never dumps full instruction into search bar

---

## 📊 Training Results

After 20 training episodes:

| Metric | Value |
|--------|-------|
| Success Rate | 70%+ |
| Avg Reward | 6.02 |
| Avg Steps | 9.6 |
| Best Reward | 11.80 |
| Tasks Covered | Google, YouTube, Wikipedia, GitHub, PyPI |

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

# Install Ollama (free local AI — no API key needed!)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b

# Set API keys for faster performance (optional)
cp .env.example .env
# Edit .env with your Groq and Gemini API keys
```

---

## ▶️ Running

```bash
# Start Ollama (local AI brain)
ollama serve &

# Launch Chat UI
python ui/chat_ui.py

# Run training loop
python train.py

# View training dashboard
python dashboard.py

# Run demo script
python demo.py
```

---

## 🎮 Example Tasks

```
Simple navigation:
"Go to github.com"
"Navigate to huggingface.co"

Search tasks:
"Search for PyTorch tutorials on Google"
"Search for machine learning on YouTube"
"Search for reinforcement learning on Wikipedia"

Multi-step tasks:
"Go to github.com/search and search for browser-use"
"Go to pypi.org and search for playwright"
"Go to npmjs.com and search for axios"
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

## 🤖 AI Brain System

```
Step 1: Groq Vision ⚡👁️ (llama-4-scout — sees screenshot, fastest)
    ↓ if rate limited
Step 2: Gemini Vision 🌟👁️ (sees screenshot, fast)
    ↓ if rate limited  
Step 3: Groq Text ⚡ (text only, fast)
    ↓ if rate limited
Step 4: Gemini Text 🌟 (text only)
    ↓ if all APIs fail
Step 5: Ollama 🦙 (local, unlimited, always works!)
```

---

## 🧠 How Few-Shot Learning Works

```
Episode 1: Search PyTorch on Google
→ Agent tries → succeeds → saved to SQLite ✅

Episode 5: Search machine learning on Google
→ Agent reads Episode 1 from DB as example
→ Already knows: click textarea[name='q'] → type → submit
→ Completes faster! ✅

Episode 20: Any search task
→ Agent has 19 examples to learn from
→ Almost always succeeds! 🎉
```

---

## 📁 Project Structure

```
Browser_Assistant/
├── env/
│   └── browser_env.py       # Core OpenEnv RL Environment + Session support
├── agent/
│   └── agent.py             # Vision AI brain + Few-shot learning
├── tasks/
│   └── task_config.py       # Smart task parser + URL builder
├── memory/
│   └── replay_buffer.py     # SQLite training memory
├── ui/
│   └── chat_ui.py           # Gradio Chat UI
├── train.py                 # Training loop
├── dashboard.py             # Training metrics dashboard
├── demo.py                  # Demo script
├── .env.example             # API keys template
└── requirements.txt         # Dependencies
```

---


---

## 🔗 Links

- [GitHub Repository](https://github.com/Yashvardhan2007/Browser_Assistant)
- [Hugging Face Demo](#) ← Coming soon!
