"""
Long-Term Persistent Memory
============================
Stores durable facts across sessions. Inspired by Hermes' memory tool.

Key design: memory is injected into every reasoning turn, so it must be
compact and focused on facts that will still matter later.

Priority hierarchy:
1. User preferences and corrections (prevent future mistakes)
2. Stable investment strategies and rules
3. Environment facts (API quirks, account details)
4. Lessons learned from past decisions

Do NOT store:
- Task progress or session state (use trade_journal instead)
- Raw data dumps (use research_corpus instead)
- Temporary observations (use session context)
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class MemoryCategory(Enum):
    USER_PREFERENCE = "user_preference"      # "I prefer dividend stocks"
    CORRECTION = "correction"                 # "Don't sell NVDA on dips"
    STRATEGY = "strategy"                     # "Always hedge tech with financials"
    ENVIRONMENT = "environment"               # "IBKR port 7497 is paper"
    LESSON = "lesson"                         # "Earnings surprises often reverse in 3 days"
    SKILL = "skill"                           # "How to interpret P/E in high-growth context"


@dataclass
class MemoryEntry:
    content: str
    category: MemoryCategory
    importance: int = 5          # 1-10, higher = more important
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        d["category"] = MemoryCategory(d["category"])
        return cls(**d)


class LongTermMemory:
    """
    Persistent memory store backed by SQLite.
    
    Usage:
        memory = LongTermMemory("data/memory.db")
        
        # Save a lesson
        memory.add(
            content="User prefers long-term holds over day trading",
            category=MemoryCategory.USER_PREFERENCE,
            importance=8,
            tags=["trading_style", "preference"]
        )
        
        # Retrieve relevant memories for current context
        context_memories = memory.get_relevant(
            context="deciding whether to sell AAPL",
            max_entries=5
        )
        
        # Get all memories as injectable text
        memory_text = memory.get_injection_text()
    """
    
    MAX_INJECTION_CHARS = 2000  # Keep injection compact
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    tags TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_category 
                ON memories(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON memories(importance DESC)
            """)
            conn.commit()
    
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    def add(self, content: str, category: MemoryCategory,
            importance: int = 5, tags: List[str] = None) -> int:
        """Add a new memory entry. Returns the entry ID."""
        entry = MemoryEntry(
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
        )
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO memories 
                   (content, category, importance, tags, created_at, last_accessed, access_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (entry.content, entry.category.value, entry.importance,
                 json.dumps(entry.tags), entry.created_at, entry.last_accessed)
            )
            conn.commit()
            return cursor.lastrowid
    
    def add_correction(self, content: str, importance: int = 9, tags: List[str] = None):
        """Shortcut: add a user correction (highest priority)."""
        return self.add(content, MemoryCategory.CORRECTION, importance, tags)
    
    def add_preference(self, content: str, importance: int = 7, tags: List[str] = None):
        """Shortcut: add a user preference."""
        return self.add(content, MemoryCategory.USER_PREFERENCE, importance, tags)
    
    def add_lesson(self, content: str, importance: int = 6, tags: List[str] = None):
        """Shortcut: add a lesson learned."""
        return self.add(content, MemoryCategory.LESSON, importance, tags)
    
    def add_strategy(self, content: str, importance: int = 7, tags: List[str] = None):
        """Shortcut: add a strategy rule."""
        return self.add(content, MemoryCategory.STRATEGY, importance, tags)
    
    def get_all(self, category: Optional[MemoryCategory] = None,
                include_archived: bool = False) -> List[MemoryEntry]:
        """Retrieve all memories, optionally filtered by category."""
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
        if not include_archived:
            query += " AND archived = 0"
        
        query += " ORDER BY importance DESC, created_at DESC"
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def search(self, keywords: List[str], max_results: int = 10) -> List[MemoryEntry]:
        """Simple keyword search across memory content and tags."""
        with self._conn() as conn:
            conditions = []
            params = []
            for kw in keywords:
                conditions.append("(content LIKE ? OR tags LIKE ?)")
                params.extend([f"%{kw}%", f"%{kw}%"])
            
            query = f"""
                SELECT * FROM memories 
                WHERE archived = 0 AND ({' OR '.join(conditions)})
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """
            params.append(max_results)
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def get_injection_text(self) -> str:
        """
        Get compact text representation of all memories for injection
        into agent reasoning context. Capped at MAX_INJECTION_CHARS.
        
        This is the key integration point: call this before every
        reasoning/decision step to provide the agent with persistent context.
        """
        entries = self.get_all()
        if not entries:
            return ""
        
        lines = ["=== PERSISTENT MEMORY ==="]
        total_chars = len(lines[0])
        
        for entry in entries:
            cat_label = entry.category.value.upper().replace("_", " ")
            line = f"[{cat_label}] {entry.content}"
            
            if total_chars + len(line) + 1 > self.MAX_INJECTION_CHARS:
                lines.append(f"... ({len(entries) - lines.__len__()} more entries truncated)")
                break
            
            lines.append(line)
            total_chars += len(line) + 1
            
            # Update access tracking
            self._touch(entry)
        
        return "\n".join(lines)
    
    def archive(self, memory_id: int):
        """Archive a memory (soft delete)."""
        with self._conn() as conn:
            conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))
            conn.commit()
    
    def remove(self, memory_id: int):
        """Permanently delete a memory."""
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
    
    def clear_category(self, category: MemoryCategory):
        """Archive all memories in a category."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET archived = 1 WHERE category = ?",
                (category.value,)
            )
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 0"
            ).fetchone()[0]
            
            by_category = conn.execute(
                """SELECT category, COUNT(*) FROM memories 
                   WHERE archived = 0 GROUP BY category"""
            ).fetchall()
            
            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM memories WHERE archived = 0"
            ).fetchone()[0]
        
        return {
            "total_active": total,
            "by_category": {row[0]: row[1] for row in by_category},
            "avg_importance": round(avg_importance or 0, 1),
        }
    
    def _touch(self, entry: MemoryEntry):
        """Update access tracking for a memory entry."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE memories 
                   SET last_accessed = ?, access_count = access_count + 1 
                   WHERE content = ? AND category = ?""",
                (time.time(), entry.content, entry.category.value)
            )
            conn.commit()
    
    @staticmethod
    def _row_to_entry(row) -> MemoryEntry:
        return MemoryEntry(
            content=row[1],
            category=MemoryCategory(row[2]),
            importance=row[3],
            tags=json.loads(row[4]),
            created_at=row[5],
            last_accessed=row[6],
            access_count=row[7],
        )
