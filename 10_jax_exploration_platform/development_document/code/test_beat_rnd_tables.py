"""Tests of the splicing and number formatting of beat_rnd_tables.py (no run records are read)."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_rnd_tables as tables  # noqa: E402


def test_inject_table_replaces_only_the_marked_block(tmp_path):
    """The text between one pair of markers is replaced; the caption and everything else stay."""
    tex = tmp_path / 'section.tex'
    tex.write_text('before\n% >>> AUTO-GENERATED TABLE START: a\nold\n% <<< AUTO-GENERATED TABLE END: a\n'
                   '\\caption{kept}\n')
    tables.inject_table(tex, 'a', 'new rows')
    assert tex.read_text() == ('before\n% >>> AUTO-GENERATED TABLE START: a\nnew rows\n'
                               '% <<< AUTO-GENERATED TABLE END: a\n\\caption{kept}\n')


def test_inject_table_refuses_a_missing_or_repeated_marker(tmp_path):
    """A marker that is absent or appears twice stops the generator."""
    tex = tmp_path / 'section.tex'
    tex.write_text('% >>> AUTO-GENERATED TABLE START: a\n% <<< AUTO-GENERATED TABLE END: a\n' * 2)
    with pytest.raises(SystemExit):
        tables.inject_table(tex, 'a', 'x')
    with pytest.raises(SystemExit):
        tables.inject_table(tex, 'b', 'x')


def test_number_cells():
    """Means, differences and step counts come out as LaTeX math with their marks."""
    assert tables.math('-1,784') == '$-1{,}784$'
    assert tables.mean_cell(0.94041, 0.00302, 4) == r'$0.9404$ \newline $\pm 0.0030$'
    found = {'mean': -0.0003, 'interval': [-0.0067, 0.0059]}
    assert tables.difference_cell(found, 4, 'best') == r'{\bfseries\boldmath $-0.0003$} \newline $[-0.0067, +0.0059]$'
    assert tables.difference_text(found) == '$-0.0003$ $[-0.0067, +0.0059]$'
    assert tables.styled('x', 'second') == r'\underline{x}'
    assert tables.round_steps(8388608) == '8.39M'
    assert tables.round_steps(1048576) == '1.05M'
