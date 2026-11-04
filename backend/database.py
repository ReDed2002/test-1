import sqlite3

class HistoryDatabase:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sign TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    recognized_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS sentence_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signs TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    saved_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """)

    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def add(self, sign, translation, confidence):
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO recognition_history (sign, translation, confidence) VALUES (?, ?, ?)",
                (sign, translation, confidence)
            )
            return cur.lastrowid

    def recent(self, limit=100):
        with self.connect() as con:
            rows = con.execute(
                "SELECT id, sign, translation, confidence, recognized_at FROM recognition_history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def clear(self):
        with self.connect() as con:
            con.execute("DELETE FROM recognition_history")

    def add_sentence(self, signs, translation):
        import json
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO sentence_history (signs, translation) VALUES (?, ?)",
                (json.dumps(signs, ensure_ascii=False), translation)
            )
            return cur.lastrowid

    def recent_sentences(self, limit=50):
        import json
        with self.connect() as con:
            rows = con.execute(
                "SELECT id, signs, translation, saved_at FROM sentence_history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{**dict(row), "signs": json.loads(row["signs"])} for row in rows]
