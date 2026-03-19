import sqlite3
import json
import os
from datetime import datetime

# ============================================================
# PASTE THIS IN: memory/replay_buffer.py
# SQLite database for storing all training experience
# ============================================================

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "browsergym.db")


class ReplayBuffer:
    """
    SQLite-based memory for the RL agent.
    Stores every episode and step so the agent can learn from past experience.
    """

    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Episodes table — one row per complete task attempt
        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                task_type TEXT,
                total_reward REAL,
                steps_taken INTEGER,
                success INTEGER,
                final_url TEXT,
                brain_used TEXT,
                timestamp TEXT
            )
        """)

        # Steps table — one row per action taken
        c.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step_number INTEGER,
                action_type TEXT,
                selector TEXT,
                value TEXT,
                reward REAL,
                url_before TEXT,
                url_after TEXT,
                brain_used TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            )
        """)

        # Training stats table — aggregated stats per episode
        c.execute("""
            CREATE TABLE IF NOT EXISTS training_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_number INTEGER,
                avg_reward REAL,
                success_rate REAL,
                avg_steps REAL,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ── Save a complete episode ────────────────────────────
    def save_episode(self, task: str, task_type: str, total_reward: float,
                     steps_taken: int, success: bool, final_url: str,
                     brain_used: str, steps_log: list) -> int:
        """Save a complete episode to the database. Returns episode_id."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Save episode
        c.execute("""
            INSERT INTO episodes (task, task_type, total_reward, steps_taken, success, final_url, brain_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (task, task_type, total_reward, steps_taken, int(success),
              final_url, brain_used, datetime.now().isoformat()))

        episode_id = c.lastrowid

        # Save each step
        for step in steps_log:
            c.execute("""
                INSERT INTO steps (episode_id, step_number, action_type, selector, value, reward, url_before, url_after, brain_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (episode_id, step.get("step"), step.get("action"),
                  step.get("selector", ""), step.get("value", ""),
                  step.get("reward"), step.get("url_before", ""),
                  step.get("url_after", ""), step.get("brain", "")))

        # Update training stats
        self._update_stats(c, episode_id)

        conn.commit()
        conn.close()
        return episode_id

    def _update_stats(self, c, latest_episode_id):
        """Recalculate and save training stats."""
        c.execute("SELECT COUNT(*), AVG(total_reward), AVG(steps_taken), AVG(success) FROM episodes")
        row = c.fetchone()
        total_episodes, avg_reward, avg_steps, success_rate = row

        c.execute("""
            INSERT INTO training_stats (episode_number, avg_reward, success_rate, avg_steps, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (total_episodes, avg_reward or 0, success_rate or 0,
              avg_steps or 0, datetime.now().isoformat()))

    # ── Query methods ──────────────────────────────────────
    def get_all_episodes(self) -> list:
        """Get all episodes for plotting."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM episodes ORDER BY id ASC")
        rows = c.fetchall()
        conn.close()
        return rows

    def get_training_stats(self) -> list:
        """Get training stats for reward graph."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT episode_number, avg_reward, success_rate, avg_steps FROM training_stats ORDER BY id ASC")
        rows = c.fetchall()
        conn.close()
        return rows

    def get_best_actions(self, task_type: str) -> list:
        """Get the most successful actions for a given task type."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT s.action_type, s.selector, s.value, AVG(s.reward) as avg_reward, COUNT(*) as count
            FROM steps s
            JOIN episodes e ON s.episode_id = e.id
            WHERE e.task_type = ? AND e.success = 1
            GROUP BY s.action_type, s.selector, s.value
            ORDER BY avg_reward DESC
            LIMIT 10
        """, (task_type,))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_summary(self) -> dict:
        """Get overall training summary."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM episodes")
        total_episodes = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM episodes WHERE success = 1")
        successful = c.fetchone()[0]

        c.execute("SELECT AVG(total_reward) FROM episodes")
        avg_reward = c.fetchone()[0] or 0

        c.execute("SELECT AVG(steps_taken) FROM episodes")
        avg_steps = c.fetchone()[0] or 0

        c.execute("SELECT MAX(total_reward) FROM episodes")
        best_reward = c.fetchone()[0] or 0

        c.execute("SELECT task, total_reward FROM episodes ORDER BY total_reward DESC LIMIT 1")
        best_episode = c.fetchone()

        conn.close()

        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful,
            "success_rate": (successful / total_episodes * 100) if total_episodes > 0 else 0,
            "avg_reward": round(avg_reward, 2),
            "avg_steps": round(avg_steps, 2),
            "best_reward": round(best_reward, 2),
            "best_task": best_episode[0] if best_episode else "None",
        }

    def clear_all(self):
        """Clear all data — start fresh."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM steps")
        c.execute("DELETE FROM episodes")
        c.execute("DELETE FROM training_stats")
        conn.commit()
        conn.close()