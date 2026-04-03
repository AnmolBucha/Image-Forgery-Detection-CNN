import sqlite3
import os
from datetime import datetime
from config import Config


class Database:
    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(__file__), "..", "forgery_detection.db"
        )
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                num_patches INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                total_images INTEGER DEFAULT 0,
                forged_count INTEGER DEFAULT 0,
                authentic_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_patches REAL DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_images INTEGER DEFAULT 0,
                forged_count INTEGER DEFAULT 0,
                authentic_count INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def add_analysis(
        self, filename, prediction, confidence, num_patches, session_id=None
    ):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analysis_history (filename, prediction, confidence, num_patches, session_id)
            VALUES (?, ?, ?, ?, ?)
        """,
            (filename, prediction, confidence, num_patches, session_id),
        )

        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (today,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """
                UPDATE daily_stats 
                SET total_images = total_images + 1,
                    forged_count = forged_count + ?,
                    authentic_count = authentic_count + ?,
                    avg_confidence = (avg_confidence * total_images + ?) / (total_images + 1),
                    avg_patches = (avg_patches * total_images + ?) / (total_images + 1)
                WHERE date = ?
            """,
                (
                    1 if prediction == "Forged" else 0,
                    1 if prediction == "Authentic" else 0,
                    confidence,
                    num_patches,
                    today,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO daily_stats (date, total_images, forged_count, authentic_count, avg_confidence, avg_patches)
                VALUES (?, 1, ?, ?, ?, ?)
            """,
                (
                    today,
                    1 if prediction == "Forged" else 0,
                    1 if prediction == "Authentic" else 0,
                    confidence,
                    num_patches,
                ),
            )

        conn.commit()
        conn.close()
        return True

    def get_history(self, limit=50, offset=0):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM analysis_history 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """,
            (limit, offset),
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_history_count(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM analysis_history")
        count = cursor.fetchone()["count"]
        conn.close()
        return count

    def get_daily_stats(self, days=7):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM daily_stats 
            ORDER BY date DESC 
            LIMIT ?
        """,
            (days,),
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_overall_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total_analyses,
                SUM(CASE WHEN prediction = 'Forged' THEN 1 ELSE 0 END) as total_forged,
                SUM(CASE WHEN prediction = 'Authentic' THEN 1 ELSE 0 END) as total_authentic,
                AVG(confidence) as avg_confidence,
                AVG(num_patches) as avg_patches,
                MAX(confidence) as max_confidence,
                MIN(confidence) as min_confidence
            FROM analysis_history
        """)

        stats = dict(cursor.fetchone())
        conn.close()
        return stats

    def get_weekly_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                strftime('%w', timestamp) as day,
                COUNT(*) as count,
                SUM(CASE WHEN prediction = 'Forged' THEN 1 ELSE 0 END) as forged
            FROM analysis_history
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY strftime('%w', timestamp)
        """)

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_hourly_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count,
                SUM(CASE WHEN prediction = 'Forged' THEN 1 ELSE 0 END) as forged
            FROM analysis_history
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """)

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def clear_history(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history")
        cursor.execute("DELETE FROM daily_stats")
        cursor.execute("DELETE FROM session_stats")
        conn.commit()
        conn.close()
        return True

    def delete_history_item(self, item_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
        return True


db = Database()
