import sqlite3
import os
from datetime import datetime

DB_PATH = "conversation_memory.db"

class MemoryManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if "/" in DB_PATH else None
        self.conn = sqlite3.connect(DB_PATH)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_message(self, session_id, role, content):
        self.conn.execute(
            "INSERT INTO memory (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_history(self, session_id, limit=5):
        cursor = self.conn.execute(
            "SELECT role, content FROM memory WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        # Return in chronological order (oldest → newest)
        return list(reversed(rows))

    def clear_memory(self, session_id):
        self.conn.execute("DELETE FROM memory WHERE session_id=?", (session_id,))
        self.conn.commit()
