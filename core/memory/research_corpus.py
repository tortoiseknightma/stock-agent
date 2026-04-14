"""
Research Corpus Manager
========================
Indexed collection of research reports, earnings transcripts, SEC filings,
and analyst notes. Uses FTS5 full-text search for retrieval.

Key design (from Hermes):
- Research is indexed but NOT injected into memory automatically
- Agent queries the corpus on-demand when reasoning about specific tickers
- Prevents information overload while maintaining access to deep context
- Supports tagging by ticker, sector, date, source type

The corpus acts as the agent's "reference library" — always available
but only consulted when relevant.
"""

import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class DocType(Enum):
    RESEARCH_REPORT = "research_report"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    SEC_FILING = "sec_filing"
    NEWS_ARTICLE = "news_article"
    ANALYST_NOTE = "analyst_note"
    INDUSTRY_REPORT = "industry_report"
    ECONOMIC_DATA = "economic_data"
    AGENT_ANALYSIS = "agent_analysis"    # StockAgent's own analysis outputs


@dataclass
class ResearchDoc:
    title: str
    content: str
    doc_type: DocType
    tickers: List[str] = field(default_factory=list)        # Related stock tickers
    sectors: List[str] = field(default_factory=list)         # Industry sectors
    source: str = ""                                         # e.g., "Goldman Sachs", "SEC EDGAR"
    url: str = ""
    published_at: Optional[float] = None                     # Original publication timestamp
    ingested_at: float = field(default_factory=time.time)
    summary: str = ""                                        # LLM-generated summary
    tags: List[str] = field(default_factory=list)
    relevance_score: float = 0.0                             # Updated by access patterns
    
    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["doc_type"] = self.doc_type.value
        d["content_hash"] = self.content_hash
        return d


class ResearchCorpus:
    """
    Searchable research document corpus backed by SQLite FTS5.
    
    Usage:
        corpus = ResearchCorpus("data/research_corpus")
        
        # Ingest a research report
        corpus.add(ResearchDoc(
            title="AAPL Q4 2025 Earnings Analysis",
            content="Apple reported...",
            doc_type=DocType.RESEARCH_REPORT,
            tickers=["AAPL"],
            sectors=["Technology"],
            source="Morgan Stanley"
        ))
        
        # Search for relevant context when analyzing AAPL
        docs = corpus.search("AAPL services growth", tickers=["AAPL"], limit=5)
        
        # Get all docs for a specific ticker
        aapl_docs = corpus.get_by_ticker("AAPL")
    """
    
    def __init__(self, base_path: str = "data/research_corpus"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_path / "corpus.db"
        self.docs_path = self.base_path / "docs"
        self.docs_path.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with FTS5 for full-text search."""
        with self._conn() as conn:
            # Main documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    doc_type TEXT NOT NULL,
                    tickers TEXT DEFAULT '[]',
                    sectors TEXT DEFAULT '[]',
                    source TEXT DEFAULT '',
                    url TEXT DEFAULT '',
                    published_at REAL,
                    ingested_at REAL NOT NULL,
                    summary TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    relevance_score REAL DEFAULT 0.0,
                    content_path TEXT NOT NULL,
                    archived INTEGER DEFAULT 0
                )
            """)
            
            # FTS5 virtual table for full-text search (self-contained, no external content).
            # Content is stored in files; FTS indexes title + summary + snippet of content.
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    title, summary, content, tickers, sectors, tags
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tickers 
                ON documents(tickers)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_type 
                ON documents(doc_type)
            """)
            conn.commit()
    
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    def add(self, doc: ResearchDoc) -> int:
        """
        Add a document to the corpus.
        Content is stored as a separate file to keep the DB lean.
        Returns document ID.
        """
        # Check for duplicates
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?",
                (doc.content_hash,)
            ).fetchone()
            if existing:
                return existing[0]
        
        # Store content as file
        content_file = self.docs_path / f"{doc.content_hash}.txt"
        content_file.write_text(doc.content)
        
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO documents 
                   (title, content_hash, doc_type, tickers, sectors, source, url,
                    published_at, ingested_at, summary, tags, relevance_score, content_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?)""",
                (doc.title, doc.content_hash, doc.doc_type.value,
                 json.dumps(doc.tickers), json.dumps(doc.sectors),
                 doc.source, doc.url, doc.published_at, doc.ingested_at,
                 doc.summary, json.dumps(doc.tags), str(content_file))
            )
            conn.commit()
            
            # Index in FTS5 (content snippet for search; full text in file)
            doc_id = cursor.lastrowid
            conn.execute(
                """INSERT INTO documents_fts(rowid, title, summary, content, tickers, sectors, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, doc.title, doc.summary, doc.content[:4000],
                 json.dumps(doc.tickers), json.dumps(doc.sectors),
                 json.dumps(doc.tags))
            )
            conn.commit()
            
            return doc_id
    
    def search(self, query: str, tickers: List[str] = None,
               doc_types: List[DocType] = None, limit: int = 10) -> List[ResearchDoc]:
        """
        Full-text search across the corpus.
        Optionally filter by tickers and document types.
        """
        with self._conn() as conn:
            # Use FTS5 for text search
            sql = """
                SELECT d.* FROM documents d
                JOIN documents_fts fts ON d.id = fts.rowid
                WHERE documents_fts MATCH ?
                AND d.archived = 0
            """
            params = [query]
            
            if tickers:
                ticker_conditions = []
                for t in tickers:
                    ticker_conditions.append("d.tickers LIKE ?")
                    params.append(f"%{t}%")
                sql += f" AND ({' OR '.join(ticker_conditions)})"
            
            if doc_types:
                type_conditions = []
                for dt in doc_types:
                    type_conditions.append("d.doc_type = ?")
                    params.append(dt.value)
                sql += f" AND ({' OR '.join(type_conditions)})"
            
            sql += " ORDER BY d.relevance_score DESC, d.ingested_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
        
        return [self._row_to_doc(row) for row in rows]
    
    def get_by_ticker(self, ticker: str, limit: int = 20) -> List[ResearchDoc]:
        """Get all documents related to a specific ticker."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM documents 
                   WHERE tickers LIKE ? AND archived = 0
                   ORDER BY relevance_score DESC, ingested_at DESC
                   LIMIT ?""",
                (f"%{ticker}%", limit)
            ).fetchall()
        
        return [self._row_to_doc(row) for row in rows]
    
    def get_by_sector(self, sector: str, limit: int = 20) -> List[ResearchDoc]:
        """Get all documents related to a specific sector."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM documents 
                   WHERE sectors LIKE ? AND archived = 0
                   ORDER BY relevance_score DESC, ingested_at DESC
                   LIMIT ?""",
                (f"%{sector}%", limit)
            ).fetchall()
        
        return [self._row_to_doc(row) for row in rows]
    
    def get_recent(self, days: int = 7, limit: int = 20) -> List[ResearchDoc]:
        """Get recently ingested documents."""
        cutoff = time.time() - (days * 86400)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM documents 
                   WHERE ingested_at > ? AND archived = 0
                   ORDER BY ingested_at DESC LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
        
        return [self._row_to_doc(row) for row in rows]
    
    def update_relevance(self, doc_id: int, delta: float = 0.1):
        """Adjust relevance score based on access patterns."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE documents SET relevance_score = relevance_score + ? WHERE id = ?",
                (delta, doc_id)
            )
            conn.commit()
    
    def get_context_for_ticker(self, ticker: str, max_chars: int = 5000) -> str:
        """
        Build a concise research context string for a ticker.
        Used to augment the agent's reasoning when analyzing specific stocks.
        """
        docs = self.get_by_ticker(ticker, limit=5)
        if not docs:
            return f"[No research corpus entries for {ticker}]"
        
        lines = [f"=== RESEARCH CONTEXT: {ticker} ==="]
        remaining = max_chars - len(lines[0])
        
        for doc in docs:
            prefix = f"[{doc.doc_type.value}] {doc.title}"
            if doc.summary:
                entry = f"{prefix}: {doc.summary}"
            else:
                entry = f"{prefix}: {doc.content[:200]}..."
            
            if len(entry) + 1 > remaining:
                break
            
            lines.append(entry)
            remaining -= len(entry) + 1
            self.update_relevance(doc.id if hasattr(doc, 'id') else 0, 0.05)
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE archived = 0"
            ).fetchone()[0]
            
            by_type = conn.execute(
                """SELECT doc_type, COUNT(*) FROM documents 
                   WHERE archived = 0 GROUP BY doc_type"""
            ).fetchall()
            
            by_ticker = conn.execute(
                "SELECT tickers FROM documents WHERE archived = 0"
            ).fetchall()
        
        # Flatten tickers
        ticker_counts = {}
        for row in by_ticker:
            for t in json.loads(row[0]):
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
        
        return {
            "total_documents": total,
            "by_type": {row[0]: row[1] for row in by_type},
            "top_tickers": dict(sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]),
        }
    
    def _row_to_doc(self, row) -> ResearchDoc:
        """Convert a database row to a ResearchDoc.

        Column order (matches CREATE TABLE):
        0:id, 1:title, 2:content_hash, 3:doc_type, 4:tickers, 5:sectors,
        6:source, 7:url, 8:published_at, 9:ingested_at, 10:summary, 11:tags,
        12:relevance_score, 13:content_path, 14:archived
        """
        content_path_val = row[13]  # content_path
        if content_path_val:
            content_path = Path(str(content_path_val))
            content = content_path.read_text(encoding="utf-8") if content_path.exists() else "[content lost]"
        else:
            content = "[content lost]"

        doc = ResearchDoc(
            title=row[1],
            content=content,
            doc_type=DocType(row[3]),
            tickers=json.loads(row[4]),
            sectors=json.loads(row[5]),
            source=row[6],
            url=row[7],
            published_at=row[8],
            ingested_at=row[9],
            summary=row[10],
            tags=json.loads(row[11]),
            relevance_score=float(row[12]) if row[12] is not None else 0.0,
        )
        doc._id = row[0]  # Store ID for updates
        return doc
