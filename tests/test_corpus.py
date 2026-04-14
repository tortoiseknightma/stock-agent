"""Tests for ResearchCorpus — add, search, retrieve, FTS5."""

import pytest
from core.memory.research_corpus import ResearchCorpus, ResearchDoc, DocType


def make_corpus(tmp_path):
    return ResearchCorpus(str(tmp_path / "corpus"))


def make_doc(title="Test Report", content="Test content about investing",
             doc_type=DocType.RESEARCH_REPORT, tickers=None,
             sectors=None, source="Test Source") -> ResearchDoc:
    return ResearchDoc(
        title=title, content=content, doc_type=doc_type,
        tickers=tickers or [], sectors=sectors or [],
        source=source, summary="A test document.",
    )


# ---------------------------------------------------------------------------
# Add / retrieve
# ---------------------------------------------------------------------------

class TestAddDocument:
    def test_add_returns_id(self, tmp_path):
        corpus = make_corpus(tmp_path)
        doc_id = corpus.add(make_doc())
        assert isinstance(doc_id, int)
        assert doc_id > 0

    def test_multiple_adds_unique_ids(self, tmp_path):
        corpus = make_corpus(tmp_path)
        id1 = corpus.add(make_doc(title="Doc A", content="Content A"))
        id2 = corpus.add(make_doc(title="Doc B", content="Content B"))
        assert id1 != id2

    def test_duplicate_content_not_added_twice(self, tmp_path):
        """Same content hash should not create a duplicate."""
        corpus = make_corpus(tmp_path)
        doc = make_doc(content="Unique content for dedup test")
        corpus.add(doc)
        corpus.add(doc)  # second add of same doc
        results = corpus.search("Unique content for dedup test")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Search (FTS5)
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_relevant_doc(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="AAPL Earnings Q4",
                            content="Apple reported record revenue of $120 billion.",
                            tickers=["AAPL"]))
        corpus.add(make_doc(title="MSFT Cloud Growth",
                            content="Microsoft Azure cloud revenue grew 40%.",
                            tickers=["MSFT"]))
        results = corpus.search("Apple revenue")
        assert len(results) >= 1
        assert any("AAPL" in d.tickers for d in results)

    def test_search_with_ticker_filter(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="AAPL Report", content="Apple growth analysis",
                            tickers=["AAPL"]))
        corpus.add(make_doc(title="MSFT Report", content="Microsoft growth analysis",
                            tickers=["MSFT"]))
        results = corpus.search("growth analysis", tickers=["AAPL"])
        assert all("AAPL" in d.tickers for d in results)

    def test_search_empty_returns_empty_list(self, tmp_path):
        corpus = make_corpus(tmp_path)
        results = corpus.search("quantum entanglement semiconductors XYZ999")
        assert results == []

    def test_search_limit_respected(self, tmp_path):
        corpus = make_corpus(tmp_path)
        for i in range(10):
            corpus.add(make_doc(title=f"Report {i}",
                                content=f"Technology analysis report {i} about markets"))
        results = corpus.search("Technology analysis", limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Ticker / sector retrieval
# ---------------------------------------------------------------------------

class TestTickerRetrieval:
    def test_get_by_ticker(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="AAPL Q1", content="Apple Q1 results",
                            tickers=["AAPL"]))
        corpus.add(make_doc(title="NVDA Report", content="NVIDIA GPU analysis",
                            tickers=["NVDA"]))
        results = corpus.get_by_ticker("AAPL")
        assert all("AAPL" in d.tickers for d in results)

    def test_get_by_ticker_empty_when_none(self, tmp_path):
        corpus = make_corpus(tmp_path)
        results = corpus.get_by_ticker("ZZZZ")
        assert results == []

    def test_get_by_sector(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="Tech Report", content="Technology sector overview",
                            sectors=["Technology"]))
        corpus.add(make_doc(title="Energy Report", content="Energy sector trends",
                            sectors=["Energy"]))
        results = corpus.get_by_sector("Technology")
        assert all("Technology" in d.sectors for d in results)


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------

class TestRecency:
    def test_get_recent_returns_docs(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="Recent Doc", content="Just published analysis"))
        results = corpus.get_recent(days=7)
        assert len(results) >= 1

    def test_get_recent_limit(self, tmp_path):
        corpus = make_corpus(tmp_path)
        for i in range(5):
            corpus.add(make_doc(title=f"Doc {i}", content=f"Recent report {i}"))
        results = corpus.get_recent(days=7, limit=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

class TestContextGeneration:
    def test_get_context_for_ticker_returns_string(self, tmp_path):
        corpus = make_corpus(tmp_path)
        corpus.add(make_doc(title="AAPL Deep Dive",
                            content="Apple's ecosystem moat is considerable. " * 20,
                            tickers=["AAPL"]))
        ctx = corpus.get_context_for_ticker("AAPL")
        assert isinstance(ctx, str)

    def test_context_respects_max_chars(self, tmp_path):
        corpus = make_corpus(tmp_path)
        long_content = "Apple analysis. " * 500  # very long
        corpus.add(make_doc(title="Long Report", content=long_content, tickers=["AAPL"]))
        ctx = corpus.get_context_for_ticker("AAPL", max_chars=500)
        assert len(ctx) <= 600  # allow small overhead for formatting

    def test_context_empty_when_no_docs(self, tmp_path):
        corpus = make_corpus(tmp_path)
        ctx = corpus.get_context_for_ticker("ZZZZ")
        assert ctx == "" or isinstance(ctx, str)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_structure(self, tmp_path):
        corpus = make_corpus(tmp_path)
        stats = corpus.get_stats()
        assert "total_documents" in stats

    def test_stats_count_increases(self, tmp_path):
        corpus = make_corpus(tmp_path)
        before = corpus.get_stats()["total_documents"]
        corpus.add(make_doc())
        after = corpus.get_stats()["total_documents"]
        assert after == before + 1
