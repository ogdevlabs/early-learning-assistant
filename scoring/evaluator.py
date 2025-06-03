import sqlite3
from datetime import datetime
from typing import Tuple

DB_PATH = "scoring.db"

class ScoringEvaluator:
    def __init__(self):
        """Create a SQLite database and a table for storing scores."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
               CREATE TABLE IF NOT EXISTS scoring_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    facial_point TEXT NOT NULL,
                    success BOOLEAN NOT NULL
               )
        ''')
        conn.commit()
        conn.close()

    def log_scoring_event(self, user_id: str, facial_point: str, success: bool):
        """Log a scoring event to the database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scoring_events (user_id, facial_point, success)
            VALUES (?, ?, ?)
        ''', (user_id, facial_point, success))
        conn.commit()
        conn.close()
        print(f"Logged scoring event: user_id={user_id}, facial_point={facial_point}, success={success}")