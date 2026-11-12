"""Tests of the report's contents table and its read / unread / updated marking."""

import json

from maze_campaign.analysis import contents

REPORT = '# Title\n\nGenerated now.\n\n## Measures\n\nW is the mean.\n\n## Results: first table\n\nW is 0.31 at $B$.\n\n'


def states(analysis_dir):
    """Each section's state as the manifest has it."""
    history = contents.read_history(analysis_dir)
    return {name: contents.status_of(history['sections'][name]) for name in history['order']}


def test_sections_start_unread_and_become_read_only_when_marked(tmp_path):
    """A new report's sections are unread; marking makes them read; regenerating the same text keeps them read."""
    assert contents.write_report(tmp_path, REPORT, 'Title') == {'measures': 'unread', 'results-first-table': 'unread'}
    text = (tmp_path / 'analysis.md').read_text()
    assert text.startswith('# Title\n\nGenerated now.\n\n## Contents\n') and '| [Measures](#measures) |' in text
    assert text.count('| unread |') == 2 and ' PT | ' in text and text.endswith('W is 0.31 at $B$.\n\n')
    page = (tmp_path / 'analysis.html').read_text()
    assert '<section id="measures" class="unread">' in page and 'class="unread" href="#measures"' in page
    assert contents.mark_read(tmp_path, ['measures'], 'Title') == ['measures']
    assert states(tmp_path) == {'measures': 'read', 'results-first-table': 'unread'}
    contents.write_report(tmp_path, REPORT.replace('Generated now', 'Generated later'), 'Title')
    assert states(tmp_path) == {'measures': 'read', 'results-first-table': 'unread'}


def test_a_changed_section_is_updated_and_only_its_changed_words_are_marked(tmp_path):
    """After a read, a changed number makes the section updated; the page colours the new number only."""
    contents.write_report(tmp_path, REPORT, 'Title')
    contents.mark_read(tmp_path, ['all'], 'Title')
    first = contents.read_history(tmp_path)['sections']['results-first-table']
    contents.write_report(tmp_path, REPORT.replace('0.31', '0.35'), 'Title')
    assert states(tmp_path) == {'measures': 'read', 'results-first-table': 'updated'}
    record = contents.read_history(tmp_path)['sections']['results-first-table']
    assert record['first_added'] == first['first_added'] and record['last_changed'] >= first['last_changed']
    assert record['last_read_text'] == '\nW is 0.31 at $B$.\n\n'
    page = (tmp_path / 'analysis.html').read_text()
    assert page.count('<span class="changed">') == 1 and '<span class="changed">0.35</span>' in page
    assert '<section id="results-first-table" class="updated">' in page
    assert '| updated |' in (tmp_path / 'analysis.md').read_text()


def test_a_changed_figure_makes_its_section_updated_and_gets_a_label(tmp_path):
    """A figure whose image file changed since the last read is labelled, though the text is the same."""
    (tmp_path / 'plots').mkdir()
    (tmp_path / 'plots/curve.png').write_bytes(b'first image')
    report = REPORT + '## Curves\n\n![curve](plots/curve.png)\n\n'
    contents.write_report(tmp_path, report, 'Title')
    contents.mark_read(tmp_path, ['all'], 'Title')
    (tmp_path / 'plots/curve.png').write_bytes(b'second image')
    assert contents.write_report(tmp_path, report, 'Title')['curves'] == 'updated'
    page = (tmp_path / 'analysis.html').read_text()
    assert '<small class="figure-updated">updated</small>' in page and '<span class="changed">' not in page


def test_changed_ranges_compare_words_and_keep_a_formula_whole():
    """A changed number is one range; a changed formula is marked from dollar sign to dollar sign."""
    assert contents.changed_ranges('W is 0.31 at $B$', 'W is 0.35 at $B$') == [(5, 9)]
    assert contents.changed_ranges('at $B$ steps', 'at $2 B$ steps') == [(3, 8)]
    assert contents.changed_ranges('same text', 'same text') == []


def test_section_identifiers_and_a_repeated_title(tmp_path):
    """Titles become link targets; two sections with one target are refused; an unknown section cannot be marked."""
    assert contents.section_identifier('Rollout shape: environments per policy') == 'rollout-shape-environments-per-policy'
    assert contents.split_sections('# T\n\nintro\n\n## A\ntext a\n### inner\n## B\ntext b\n') == (
        '# T\n\nintro\n\n', [('A', 'text a\n### inner\n'), ('B', 'text b\n')])
    try:
        contents.write_report(tmp_path, '# T\n\n## Same\na\n## Same\nb\n', 'T')
    except ValueError as error:
        assert 'share a link target' in str(error)
    else:
        raise AssertionError('two sections with one title were accepted')
    contents.write_report(tmp_path, REPORT, 'Title')
    try:
        contents.mark_read(tmp_path, ['no-such-section'], 'Title')
    except ValueError as error:
        assert 'No section' in str(error)
    else:
        raise AssertionError('an unknown section was marked as read')
    assert json.loads((tmp_path / contents.HISTORY_FILE).read_text())['order'] == ['measures', 'results-first-table']


def test_a_report_with_another_file_name_keeps_its_own_files(tmp_path):
    """A batch summary is written as summary.md and summary.html, and marking it read renders the same files."""
    assert contents.write_report(tmp_path, REPORT, 'Title', name='summary') == {
        'measures': 'unread', 'results-first-table': 'unread'}
    assert (tmp_path / 'summary.html').exists() and not (tmp_path / 'analysis.md').exists()
    assert 'brown](summary.html)' in (tmp_path / 'summary.md').read_text()
    contents.mark_read(tmp_path, ['all'], 'Title', name='summary')
    assert states(tmp_path) == {'measures': 'read', 'results-first-table': 'read'}
    assert '| read |' in (tmp_path / 'summary.md').read_text() and not (tmp_path / 'analysis.md').exists()
