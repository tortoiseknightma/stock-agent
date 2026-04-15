"""
Situation Memory
=================
Per-role episodic memory for agent debate roles.
Uses SQLite FTS5 for full-text retrieval of past decisions and lessons.

Each agent role (bull, bear, aggressive, conservative, neutral, etc.) gets
its own memory namespace. When building a prompt, the agent retrieves the
most relevant past situations and injects their lessons as context.

Usage::

    mem = SituationMemory("data/agent_memory.db", role="aggressive")
    mem.store(situation="AAPL Q4 earnings beat", decision="full position",
              outcome="lost 8% on macro selloff", lesson="earnings beats alone don't protect from rate shock")

    lessons = mem.retrieve("AAPL earnings beat rate risk", top_k=3)
    # → ["earnings beats alone don't protect from rate shock", ...]
"""

import sqlite3
import time
from pathlib import Path
from typing import List


class SituationMemory:
    """
    Episodic memory for a single agent role backed by SQLite FTS5.

    Each instance is scoped to one role (e.g. "bull", "aggressive").
    Multiple roles can share the same db_path without collision.
    """

    def __init__(self, db_path: str = "data/agent_memory.db", role: str = "default"):
        self._db_path = str(Path(db_path))
        self._role = role
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, situation: str, decision: str, outcome: str, lesson: str) -> None:
        """
        Store a past decision and its lesson.

        Args:
            situation: Context in which the decision was made (ticker, macro conditions, etc.)
            decision:  What the agent argued/decided
            outcome:   What actually happened (profit/loss, realized risk, etc.)
            lesson:    The distilled takeaway (injected into future prompts)
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO situations (role, situation, decision, outcome, lesson, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self._role, situation, decision, outcome, lesson, time.time()),
            )
            conn.execute(
                """
                INSERT INTO situations_fts (situation, decision, outcome, lesson)
                VALUES (?, ?, ?, ?)
                """,
                (situation, decision, outcome, lesson),
            )
            conn.commit()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant lessons for a given situation description.

        Args:
            query:  Free-text description of the current situation
            top_k:  Maximum number of lessons to return

        Returns:
            List of lesson strings, most relevant first.
        """
        if not query.strip():
            return []

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT s.lesson
                FROM situations s
                JOIN situations_fts ON situations_fts.rowid = s.rowid
                WHERE situations_fts MATCH ?
                  AND s.role = ?
                ORDER BY rank
                LIMIT ?
                """,
                (self._fts_query(query), self._role, top_k),
            ).fetchall()

        return [row[0] for row in rows]

    def count(self) -> int:
        """Return the number of stored situations for this role."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM situations WHERE role = ?", (self._role,)
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS situations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT    NOT NULL,
                    situation  TEXT    NOT NULL,
                    decision   TEXT    NOT NULL,
                    outcome    TEXT    NOT NULL,
                    lesson     TEXT    NOT NULL,
                    created_at REAL    NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS situations_fts USING fts5(
                    situation, decision, outcome, lesson
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_situations_role ON situations(role)"
            )
            conn.commit()

    @staticmethod
    def _fts_query(raw: str) -> str:
        """
        Convert a free-text query to an FTS5 query.
        Strips FTS5 special characters to prevent syntax errors.
        """
        # Remove FTS5 operators to avoid parse errors
        safe = (
            raw.replace('"', " ")
               .replace("'", " ")
               .replace("*", " ")
               .replace("(", " ")
               .replace(")", " ")
               .replace(":", " ")
               .replace("-", " ")
               .replace("+", " ")
        )
        tokens = safe.split()
        if not tokens:
            return '""'
        # Join as OR query so partial matches still return results
        return " OR ".join(tokens)
