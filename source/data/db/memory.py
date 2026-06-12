"""Memory subsystem.

Short-term memory:
  - Per-session. Keeps the last N user/bot turns verbatim (default 5).
  - Turns older than N are compressed into a rolling summary stored in
    `Chat_Sessions.summary`. Summary is regenerated when new turns roll out.

Long-term memory:
  - Global (single-user demo). Facts extracted from user messages by the LLM
    and saved in `User_Facts`. Injected into the system context on every turn.
"""
import json
import re
import sqlite3
from typing import List, Tuple

from source.data.db.db_connection import DBConnection


SHORT_TERM_KEEP = 5  # number of most-recent messages kept verbatim


class Memory:
    def __init__(self, db: DBConnection, generator=None):
        """generator is a source.generate.generate.Gemini_Generate instance.
        Optional — summarization and fact-extraction are skipped if None."""
        self.db = db
        self.generator = generator

    # -------- SHORT-TERM (per-session) --------

    def get_session_summary(self, session_id: int) -> str:
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("SELECT summary FROM Chat_Sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return (row[0] if row and row[0] else "") or ""

    def set_session_summary(self, session_id: int, summary: str):
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("UPDATE Chat_Sessions SET summary=? WHERE id=?", (summary, session_id))
        conn.commit()
        conn.close()

    def get_recent_turns(self, session_id: int, limit: int = SHORT_TERM_KEEP) -> List[Tuple[str, str]]:
        """Return last `limit` messages as [(sender, message), ...] in chronological order."""
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT sender, message FROM Chat_Messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
        return list(reversed(rows))

    def count_messages(self, session_id: int) -> int:
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Chat_Messages WHERE session_id=?", (session_id,))
        n = cur.fetchone()[0]
        conn.close()
        return n

    def get_messages_older_than_last_n(self, session_id: int, n: int) -> List[Tuple[str, str]]:
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT sender, message FROM Chat_Messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        if len(rows) <= n:
            return []
        return rows[:-n]

    def build_context(self, session_id: int) -> str:
        """Assemble the `[long-term facts][session summary][recent turns]` block
        that will be injected as system context for the next user turn."""
        blocks = []

        facts = self.get_facts()
        if facts:
            blocks.append(
                "## Thông tin về người dùng (ghi nhớ qua nhiều phiên):\n"
                + "\n".join(f"- {f}" for f in facts)
            )

        summary = self.get_session_summary(session_id)
        if summary:
            blocks.append("## Tóm tắt các phần đã trao đổi trước đó trong phiên:\n" + summary)

        turns = self.get_recent_turns(session_id, SHORT_TERM_KEEP)
        if turns:
            lines = []
            for sender, msg in turns:
                role = "Người dùng" if sender == "user" else "Trợ lý"
                lines.append(f"{role}: {msg}")
            blocks.append("## Các lượt gần nhất trong phiên:\n" + "\n".join(lines))

        return "\n\n".join(blocks).strip()

    def refresh_summary_if_needed(self, session_id: int):
        """If the session has more than SHORT_TERM_KEEP messages, regenerate
        the rolling summary from everything older than the last N."""
        if self.generator is None:
            return
        older = self.get_messages_older_than_last_n(session_id, SHORT_TERM_KEEP)
        if not older:
            return
        try:
            transcript = "\n".join(
                f"{'Người dùng' if s == 'user' else 'Trợ lý'}: {m}" for s, m in older
            )
            existing = self.get_session_summary(session_id)
            prompt = (
                "Bạn là trợ lý tóm tắt hội thoại pháp luật. "
                "Hãy tạo bản tóm tắt tiếng Việt ngắn gọn (3-5 câu) về đoạn hội thoại sau. "
                "Ghi rõ: chủ đề người dùng hỏi, kết luận chính của trợ lý, các văn bản luật/điều luật được trích dẫn nếu có. "
                "Không bịa thông tin ngoài đoạn hội thoại.\n\n"
                + (f"Bản tóm tắt trước đó (nếu có thì tiếp nối):\n{existing}\n\n" if existing else "")
                + f"Đoạn hội thoại:\n{transcript}\n\nBản tóm tắt mới:"
            )
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            model = self.generator._make_model(temperature=0.1, max_tokens=500)
            msgs = ChatPromptTemplate.from_messages([("human", "{p}")]).format_messages(p=prompt)
            summary = (model | StrOutputParser()).invoke(msgs).strip()
            if summary:
                self.set_session_summary(session_id, summary)
        except Exception as e:
            print(f"[Memory.refresh_summary] error: {type(e).__name__}: {e}", flush=True)

    # -------- LONG-TERM (global facts) --------

    def get_facts(self, limit: int = 20) -> List[str]:
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("SELECT fact_text FROM User_Facts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def add_fact(self, fact: str):
        fact = (fact or "").strip()
        if not fact:
            return
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT OR IGNORE INTO User_Facts (fact_text) VALUES (?)", (fact,))
            conn.commit()
        finally:
            conn.close()

    def delete_fact(self, fact_id: int):
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM User_Facts WHERE id=?", (fact_id,))
        conn.commit()
        conn.close()

    def list_facts(self) -> List[dict]:
        conn = self.db.Get_DB_Connection()
        cur = conn.cursor()
        cur.execute("SELECT id, fact_text, created_at FROM User_Facts ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return [{"id": r[0], "text": r[1], "created_at": str(r[2])} for r in rows]

    def extract_and_save_facts(self, user_message: str):
        """Ask the LLM to pull durable facts (name, role, ongoing legal matter)
        from the user's latest message. Skip if nothing durable."""
        if self.generator is None or not user_message.strip():
            return
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            existing = self.get_facts()
            existing_block = "\n".join(f"- {f}" for f in existing) if existing else "(chưa có)"
            prompt = (
                "Phân tích tin nhắn của người dùng sau. Trích xuất CÁC FACT DÀI HẠN về họ "
                "(nên nhớ qua nhiều phiên chat), ví dụ:\n"
                "- Tên, nghề nghiệp, vai trò\n"
                "- Vấn đề pháp lý đang theo dõi (ví dụ: đang ly hôn, tranh chấp đất đai...)\n"
                "- Sở thích trình bày (ngắn gọn, chi tiết...)\n\n"
                "BỎ QUA các thông tin nhất thời (câu hỏi hiện tại, cảm thán, chào hỏi).\n"
                "Trả về JSON array các câu ngắn tiếng Việt. Nếu không có fact mới đáng nhớ, trả về [].\n"
                "Fact đã có (đừng lặp lại):\n" + existing_block + "\n\n"
                "Tin nhắn người dùng:\n" + user_message + "\n\n"
                "JSON:"
            )
            model = self.generator._make_model(temperature=0.0, max_tokens=400)
            msgs = ChatPromptTemplate.from_messages([("human", "{p}")]).format_messages(p=prompt)
            raw = (model | StrOutputParser()).invoke(msgs).strip()
            # Strip code fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            try:
                facts = json.loads(raw)
            except Exception:
                return
            if isinstance(facts, list):
                for f in facts:
                    if isinstance(f, str):
                        self.add_fact(f)
        except Exception as e:
            print(f"[Memory.extract_and_save_facts] error: {type(e).__name__}: {e}", flush=True)
