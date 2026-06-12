import os
import sqlite3
from source.core.config import Settings


class DBConnection:
    def __init__(self, setting: Settings):
        self.setting = setting
        self.db_path = setting.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema()

    def Get_DB_Connection(self):
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_schema(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS Chat_Sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS Chat_Messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                send_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES Chat_Sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Chat_References (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                reference_content TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES Chat_Messages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS User_Facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_text TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Best-effort migration for DBs created before `summary` column existed
        try:
            cur.execute("ALTER TABLE Chat_Sessions ADD COLUMN summary TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
