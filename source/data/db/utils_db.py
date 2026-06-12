from source.data.db.db_connection import DBConnection
from source.schema.message_input import MessageInput


class DB_Utils():
    def __init__(self, db: DBConnection):
        self.db = db

    def Create_Session(self):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Chat_Sessions (create_at) VALUES (CURRENT_TIMESTAMP)"
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def Insert_Message(self, session_id, sender, message):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO Chat_Messages (session_id, sender, message, send_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (session_id, sender, message),
        )
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return message_id

    def Insert_References(self, message_id, references):
        if not references:
            return

        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        for ref in references:
            cursor.execute(
                """
                INSERT INTO Chat_References (message_id, reference_content)
                VALUES (?, ?)
                """,
                (message_id, ref),
            )
        conn.commit()
        conn.close()

    def Get_References(self, message_id):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT reference_content
            FROM Chat_References
            WHERE message_id = ?
            """,
            (message_id,),
        )
        references = [row[0] for row in cursor.fetchall()]
        conn.close()
        return references

    def Get_Session(self):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        query = """
        SELECT
            cs.id,
            cs.create_at,
            (SELECT message FROM Chat_Messages
             WHERE session_id = cs.id AND sender = 'user'
             ORDER BY send_at ASC LIMIT 1) AS first_message
        FROM Chat_Sessions cs
        WHERE EXISTS (
            SELECT 1 FROM Chat_Messages
            WHERE session_id = cs.id AND sender = 'user'
        )
        ORDER BY cs.create_at DESC
        """
        cursor.execute(query)
        sessions = cursor.fetchall()
        conn.close()
        return sessions

    def Get_History(self, session_id):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        query = """
        SELECT id, sender, message, send_at
        FROM Chat_Messages
        WHERE session_id = ?
        ORDER BY send_at ASC
        """
        cursor.execute(query, (session_id,))
        messages = cursor.fetchall()
        conn.close()
        return messages

    def Delete_Session(self, session_id):
        conn = self.db.Get_DB_Connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Chat_Sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
