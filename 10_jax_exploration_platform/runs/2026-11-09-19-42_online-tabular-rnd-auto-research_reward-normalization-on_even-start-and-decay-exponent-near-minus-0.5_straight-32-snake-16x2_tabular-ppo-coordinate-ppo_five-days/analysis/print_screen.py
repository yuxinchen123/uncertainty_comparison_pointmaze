"""Print the screen tables of analysis.md compactly (one line per idea and case)."""
import sys

for line in open(sys.argv[1]).read().splitlines():
    if line.startswith('### '):
        print(line)
    elif line.startswith('| ') and not line.startswith('| idea') and '---' not in line and '](#' not in line and 'Section' not in line:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        setting = cells[7].replace('learning rate ', 'lr ').replace('entropy ', 'ent ').replace('weight ', 'w ')
        print('  ', cells[0][:50].ljust(50), cells[1][:42].ljust(42), cells[2].ljust(7), cells[3].ljust(5),
              cells[4][:22].ljust(22), cells[5].ljust(5), cells[6][:10].ljust(10), setting)
