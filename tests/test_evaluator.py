import os
import sqlite3
import pytest
from scoring.evaluator import ScoringEvaluator, DB_PATH

@pytest.fixture(autouse=True)
def cleanup_db():
    # Remove the database before each test for isolation
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_db_creation():
    evaluator = ScoringEvaluator()
    assert os.path.exists(DB_PATH)
    # Check table exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_events';")
    assert cursor.fetchone() is not None
    conn.close()

def test_log_scoring_event():
    evaluator = ScoringEvaluator()
    evaluator.log_scoring_event('user1', 'nose', True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, facial_point, success FROM scoring_events;")
    row = cursor.fetchone()
    assert row == ('user1', 'nose', 1)  # SQLite stores bool as int
    conn.close()

