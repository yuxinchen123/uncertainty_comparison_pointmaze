"""Generate the study report and compile its development-document section on compute."""

import argparse
import os
from pathlib import Path
import re
import subprocess

from common import PLATFORM, PROJECT, PYTHON, now, write_json


def main():
    """Compile and render new tables for inspection without login-node computation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Document generation runs on an allocated compute node.')
    subprocess.run([str(PYTHON), str(Path(__file__).parent/'report.py'),
                    '--run-dir', str(args.run_dir)], check=True)
    document = PROJECT/'10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.tex'
    result = subprocess.run(['/u/sl5nw/.TinyTeX/bin/x86_64-linux/latexmk', '-cd', '-pdf',
        '-interaction=nonstopmode', '-halt-on-error', str(document)], capture_output=True, text=True)
    (args.run_dir/'logs/latex_build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Development-document compilation failed; see logs/latex_build.log.')
    auxiliary = document.with_suffix('.aux').read_text()
    pages = []
    for label in ('environments', 'training', 'swept'):
        match = re.search(r'newlabel\{tab:grid-initialization-'+label+r'\}\{\{[^}]*\}\{(\d+)\}', auxiliary)
        if match:
            page = int(match.group(1))
            output = args.run_dir/'analysis/plots'/('verification_table_'+label)
            subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-scale-to', '1800',
                            '-png', '-singlefile', str(document.with_suffix('.pdf')), str(output)], check=True)
            pages.append({'label': label, 'page': page, 'image': str(output)+'.png'})
    write_json(args.run_dir/'document_checks.json', {'at': now(), 'compiled': True, 'table_pages': pages})


if __name__ == '__main__':
    main()
