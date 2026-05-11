# src/db_service.py
import datetime
import sqlite3

DB_PATH = "rlaif_waiting_room.db"


def init_db():
    """Creates the SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS hard_negatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            deberta_quality REAL,
            deberta_component TEXT,
            deberta_stance TEXT,
            gemini_feedback TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_interaction(user_input: str, deberta_scores: dict, gemini_feedback: str):
    """
    Save likely hard negatives where the model scored an argument strongly but
    the feedback still found substantial weaknesses.
    """
    try:
        deberta_quality = float(deberta_scores.get("quality", 0))
        feedback_lower = gemini_feedback.lower()
        weakness_markers = (
            "critical weaknesses",
            "missing evidence",
            "unsupported",
            "fallacy",
            "needs stronger",
            "revise the reasoning",
        )
        has_critical_weakness = any(
            marker in feedback_lower for marker in weakness_markers
        )

        if deberta_quality > 0.50 and has_critical_weakness:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO hard_negatives
                (timestamp, user_input, deberta_quality, deberta_component, deberta_stance, gemini_feedback)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.datetime.now().isoformat(),
                    user_input,
                    deberta_quality,
                    deberta_scores.get("component", "N/A"),
                    deberta_scores.get("stance", "N/A"),
                    gemini_feedback,
                ),
            )
            conn.commit()
            conn.close()
            print("[RLAIF LOGGED] Saved a hard negative for retraining.")

    except Exception as exc:
        print(f"Database Error: {exc}")


init_db()
