"""Tests for LongTermMemory — add, retrieve, archive, injection text."""

import pytest

from core.memory.long_term import LongTermMemory, MemoryCategory, MemoryEntry


def make_memory(tmp_path):
    return LongTermMemory(str(tmp_path / "memory.db"))


class TestAddAndRetrieve:
    def test_add_returns_int_id(self, tmp_path):
        mem = make_memory(tmp_path)
        entry_id = mem.add("test content", MemoryCategory.LESSON)
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_get_all_returns_added_entries(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("prefer dividend stocks", MemoryCategory.USER_PREFERENCE)
        mem.add("earnings often reverse in 3 days", MemoryCategory.LESSON)
        entries = mem.get_all()
        assert len(entries) == 2

    def test_entries_sorted_by_importance_desc(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("low importance", MemoryCategory.LESSON, importance=2)
        mem.add("high importance", MemoryCategory.CORRECTION, importance=9)
        entries = mem.get_all()
        assert entries[0].importance >= entries[1].importance

    def test_filter_by_category(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("pref A", MemoryCategory.USER_PREFERENCE)
        mem.add("lesson B", MemoryCategory.LESSON)
        prefs = mem.get_all(category=MemoryCategory.USER_PREFERENCE)
        assert all(e.category == MemoryCategory.USER_PREFERENCE for e in prefs)
        assert len(prefs) == 1

    def test_tags_are_stored_and_retrieved(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("content", MemoryCategory.STRATEGY, tags=["tech", "hedge"])
        entries = mem.get_all()
        assert "tech" in entries[0].tags
        assert "hedge" in entries[0].tags


class TestShortcuts:
    def test_add_correction(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add_correction("don't sell NVDA on dips")
        entries = mem.get_all(category=MemoryCategory.CORRECTION)
        assert len(entries) == 1
        assert entries[0].importance == 9

    def test_add_preference(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add_preference("prefer long-term holds")
        entries = mem.get_all(category=MemoryCategory.USER_PREFERENCE)
        assert len(entries) == 1
        assert entries[0].importance == 7

    def test_add_lesson(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add_lesson("earnings surprises reverse fast")
        entries = mem.get_all(category=MemoryCategory.LESSON)
        assert len(entries) == 1

    def test_add_strategy(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add_strategy("hedge tech with financials")
        entries = mem.get_all(category=MemoryCategory.STRATEGY)
        assert len(entries) == 1


class TestSearch:
    def test_search_finds_matching_content(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("NVDA is volatile — hold long term", MemoryCategory.LESSON)
        mem.add("prefer dividend stocks", MemoryCategory.USER_PREFERENCE)
        results = mem.search(["NVDA"])
        assert len(results) == 1
        assert "NVDA" in results[0].content

    def test_search_no_match_returns_empty(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("something unrelated", MemoryCategory.LESSON)
        results = mem.search(["XYZNOTFOUND"])
        assert results == []

    def test_search_by_tag(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("content A", MemoryCategory.LESSON, tags=["earnings"])
        mem.add("content B", MemoryCategory.LESSON, tags=["dividends"])
        results = mem.search(["earnings"])
        assert len(results) == 1


class TestArchiveAndRemove:
    def test_archive_hides_entry_from_get_all(self, tmp_path):
        mem = make_memory(tmp_path)
        entry_id = mem.add("temp note", MemoryCategory.LESSON)
        mem.archive(entry_id)
        entries = mem.get_all()
        assert all(e.content != "temp note" for e in entries)

    def test_archived_included_when_flag_set(self, tmp_path):
        mem = make_memory(tmp_path)
        entry_id = mem.add("temp note", MemoryCategory.LESSON)
        mem.archive(entry_id)
        entries = mem.get_all(include_archived=True)
        assert any(e.content == "temp note" for e in entries)

    def test_remove_deletes_permanently(self, tmp_path):
        mem = make_memory(tmp_path)
        entry_id = mem.add("delete me", MemoryCategory.LESSON)
        mem.remove(entry_id)
        entries = mem.get_all(include_archived=True)
        assert all(e.content != "delete me" for e in entries)

    def test_clear_category_archives_all_in_category(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("lesson 1", MemoryCategory.LESSON)
        mem.add("lesson 2", MemoryCategory.LESSON)
        mem.add("strategy 1", MemoryCategory.STRATEGY)
        mem.clear_category(MemoryCategory.LESSON)
        remaining = mem.get_all()
        assert all(e.category != MemoryCategory.LESSON for e in remaining)
        assert any(e.category == MemoryCategory.STRATEGY for e in remaining)


class TestInjectionText:
    def test_empty_memory_returns_empty_string(self, tmp_path):
        mem = make_memory(tmp_path)
        text = mem.get_injection_text()
        assert text == ""

    def test_injection_text_contains_content(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("prefer dividend stocks", MemoryCategory.USER_PREFERENCE)
        text = mem.get_injection_text()
        assert "prefer dividend stocks" in text

    def test_injection_text_within_char_limit(self, tmp_path):
        mem = make_memory(tmp_path)
        for i in range(50):
            mem.add(f"memory entry number {i} with some filler content here",
                    MemoryCategory.LESSON)
        text = mem.get_injection_text()
        assert len(text) <= LongTermMemory.MAX_INJECTION_CHARS + 100  # small buffer for truncation line

    def test_injection_text_has_header(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("something", MemoryCategory.LESSON)
        text = mem.get_injection_text()
        assert "PERSISTENT MEMORY" in text


class TestStats:
    def test_stats_total_active(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("a", MemoryCategory.LESSON)
        mem.add("b", MemoryCategory.LESSON)
        stats = mem.get_stats()
        assert stats["total_active"] == 2

    def test_stats_by_category(self, tmp_path):
        mem = make_memory(tmp_path)
        mem.add("pref", MemoryCategory.USER_PREFERENCE)
        mem.add("lesson", MemoryCategory.LESSON)
        stats = mem.get_stats()
        assert stats["by_category"].get("user_preference") == 1
        assert stats["by_category"].get("lesson") == 1

    def test_stats_empty_db(self, tmp_path):
        mem = make_memory(tmp_path)
        stats = mem.get_stats()
        assert stats["total_active"] == 0
        assert stats["avg_importance"] == 0
