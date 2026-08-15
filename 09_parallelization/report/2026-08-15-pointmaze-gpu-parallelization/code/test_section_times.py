"""Tests for the reading-state tracking: read / unread / updated, and which blocks changed.

Run: <env python> test_section_times.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import section_times as st


def test_state():
    """A section is read only when its current digest equals the digest last marked read."""
    m = {"A": {"digest": "abc", "read_digest": "abc"},
         "B": {"digest": "def", "read_digest": "abc"},
         "C": {"digest": "ghi"}}
    assert st.state("A", m) == "read"
    assert st.state("B", m) == "updated"
    assert st.state("C", m) == "unread"
    assert st.state("missing", m) == "unread"


def test_blocks_keeps_a_table_whole():
    """Blank lines split blocks, but a table's rows stay in one block and a fence stays whole."""
    text = "para one\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n\nstill code\n```"
    b = st.blocks(text)
    assert b[0] == "para one"
    assert b[1].count("\n") == 2 and b[1].startswith("| a | b |")
    assert b[2].startswith("```") and "still code" in b[2]


def test_changed_blocks_reports_current_positions():
    """Only altered or added blocks are reported, indexed in the CURRENT text."""
    read = "A\n\nB\n\nC"
    now = "A\n\nB changed\n\nC\n\nD"
    assert st.changed_blocks(now, read) == {1, 3}
    # an unchanged document reports nothing
    assert st.changed_blocks(read, read) == set()
    # a section never read before is entirely new
    assert st.changed_blocks(read, "") == {0, 1, 2}


def test_mark_all_read_round_trip(tmp=None):
    """Marking read stores both the digest and the text, and flips the state to read."""
    with tempfile.TemporaryDirectory() as d:
        st.MANIFEST, st.SNAPSHOT = Path(d) / "m.json", Path(d) / "s.json"
        secs = {"A": "## A\n\nbody one", "B": "## B\n\nbody two"}
        m = st.mark_all_read(secs)
        assert st.state("A", m) == "read" and st.state("B", m) == "read"
        assert json.loads(st.SNAPSHOT.read_text())["A"] == "## A\n\nbody one"
        # the same sections with one changed becomes "updated", not "unread"
        m = st.stamp({"A": "## A\n\nbody one EDITED", "B": "## B\n\nbody two"})
        assert st.state("A", m) == "updated"
        assert st.state("B", m) == "read"
        assert st.changed_blocks("## A\n\nbody one EDITED", st.read_text()["A"]) == {1}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
