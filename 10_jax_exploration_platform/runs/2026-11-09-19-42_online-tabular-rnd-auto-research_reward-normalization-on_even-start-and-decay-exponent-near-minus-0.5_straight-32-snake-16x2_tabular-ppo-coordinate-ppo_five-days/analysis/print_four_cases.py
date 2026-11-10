"""Print every idea's best-setting score in the four cases, optionally only ideas whose name starts with a prefix."""
import json
import sys

rows = json.load(open(sys.argv[1]))['rows']
prefixes = tuple(sys.argv[2:])
cases = [('straight', 'tabular'), ('straight', 'neural'), ('snake', 'tabular'), ('snake', 'neural')]
table = {}
for row in rows:
    table.setdefault(row['idea'], {})[(row['kind'], row['policy'])] = row


def cell(row):
    """One case's mean capped first-goal time in thousands, with its standard error and flags."""
    if row is None:
        return '      -'.ljust(17)
    flag = ('' if row['settings_finished'] >= 28 else '*') + ('!' if row['nonfinite_copies'] else '')
    return f"{row['mean'] / 1000:8.2f}±{(row['standard_error'] or 0) / 1000:5.2f}{flag}".ljust(17)


keep = [idea for idea in table if not prefixes or idea.startswith(prefixes)]
order = sorted(keep, key=lambda idea: (sum(1 for case in cases if table[idea].get(case) is None),
                                       sum((table[idea].get(case) or {'mean': 0})['mean'] for case in cases)))
print('idea (K transitions; * = not all 28 settings finished; ! = non-finite copies)'.ljust(74),
      'straight/tabular  straight/coord    snake/tabular     snake/coord')
for idea in order:
    print(idea[:74].ljust(74), ' '.join(cell(table[idea].get(case)) for case in cases))
