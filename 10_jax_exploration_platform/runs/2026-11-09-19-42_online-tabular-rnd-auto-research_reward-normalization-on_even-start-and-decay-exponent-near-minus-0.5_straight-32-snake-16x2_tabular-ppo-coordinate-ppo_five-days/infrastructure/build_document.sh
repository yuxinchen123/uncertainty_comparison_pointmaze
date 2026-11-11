#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J rnd-auto-research-document
# Build the development document, list overfull boxes and undefined references, and render the pages of the
# schedule-fit RND subsection for inspection.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P/development_document
/u/sl5nw/.TinyTeX/bin/x86_64-linux/latexmk -cd -pdf -interaction=nonstopmode -halt-on-error RND_JAX_exploration_development_document.tex > /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/document_build.log 2>&1 || { tail -30 /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/document_build.log; exit 1; }
echo "overfull boxes: $(grep -c 'Overfull .hbox' RND_JAX_exploration_development_document.log || true)"
grep -n 'Overfull .hbox' RND_JAX_exploration_development_document.log | head -20 || true
echo "undefined references: $(grep -c 'undefined' RND_JAX_exploration_development_document.log || true)"
grep -n 'undefined' RND_JAX_exploration_development_document.log | head -8 || true
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python - <<'PY'
import fitz
doc = fitz.open('/p/rlprojects/RND/10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf')
out = '/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/analysis/document_pages'
hits = [i for i, page in enumerate(doc) if 'schedule-fit' in page.get_text() or 'Schedule-fit' in page.get_text()]
print('pages', [i + 1 for i in hits], 'of', len(doc))
for i in hits:
    doc[i].get_pixmap(dpi=100).save(f'{out}/page_{i + 1}.png')
PY
chmod g+r $P/development_document/RND_JAX_exploration_development_document.pdf
