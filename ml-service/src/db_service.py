# src/db_service.py
import sqlite3
import datetime
import os

DB_PATH = "rlaif_waiting_room.db"

def init_db():
    """Creates the SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS hard_negatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            deberta_quality REAL,
            deberta_component TEXT,
            deberta_stance TEXT,
            gemini_feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_interaction(user_input: str, deberta_scores: dict, gemini_feedback: str):
    """
    Evaluates the interaction and saves it ONLY if Gemini caught a DeBERTa mistake.
    """
    try:
        # 🧠 The "Hard Negative" Heuristic
        # If DeBERTa gave a high score, but Gemini found critical weaknesses,
        # we know the PyTorch model was fooled (e.g., by sarcasm or fake facts).
        deberta_quality = float(deberta_scores.get('quality', 0))
        
        # Check if Gemini wrote a substantial critique in the weaknesses section
        has_critical_weakness = "### ⚠️ Critical Weaknesses" in gemini_feedback
        
        # If DeBERTa thought it was > 50% quality, but Gemini flagged it, save it!
        if deberta_quality > 0.50 and has_critical_weakness:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT INTO hard_negatives 
                (timestamp, user_input, deberta_quality, deberta_component, deberta_stance, gemini_feedback)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                user_input,
                deberta_quality,
                deberta_scores.get('component', 'N/A'),
                deberta_scores.get('stance', 'N/A'),
                gemini_feedback
            ))
            conn.commit()
            conn.close()
            print("💾 [RLAIF LOGGED] Gemini corrected a DeBERTa hallucination. Saved for retraining.")
            
    except Exception as e:
        print(f"⚠️ Database Error: {e}")

# Initialize the database the first time this file is imported
init_db()