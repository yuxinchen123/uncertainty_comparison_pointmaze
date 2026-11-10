#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J rnd-fixed-document
# Regenerate the fixed-settings report (both folders), build the development document, list overfull boxes and
# render the pages that name the study's conditions.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 MPLBACKEND=Agg
cd $P/scripts/grid_initialization_campaign
/p/rlprojects/RND/.venvs/platform_jax/bin/python fixed_settings_report.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_uniform-decay-exponents-2-1-0.25_tabular-rnd-frozen-online_reward-normalization-off-on_steps-16777216_seeds-0-15_confirmation-20000-20127_baseline-to-21023
/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -m pytest -q $P/tests/test_fixed_settings_report_table.py
cd $P/development_document
/u/sl5nw/.TinyTeX/bin/x86_64-linux/latexmk -cd -pdf -interaction=nonstopmode -halt-on-error RND_JAX_exploration_development_document.tex > /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_uniform-decay-exponents-2-1-0.25_tabular-rnd-frozen-online_reward-normalization-off-on_steps-16777216_seeds-0-15_confirmation-20000-20127_baseline-to-21023/logs/latex_build.log 2>&1 || { tail -30 /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_uniform-decay-exponents-2-1-0.25_tabular-rnd-frozen-online_reward-normalization-off-on_steps-16777216_seeds-0-15_confirmation-20000-20127_baseline-to-21023/logs/latex_build.log; exit 1; }
echo "overfull boxes: $(grep -c 'Overfull .hbox' RND_JAX_exploration_development_document.log || true)"
grep -n 'Overfull .hbox' RND_JAX_exploration_development_document.log | head -20 || true
echo "undefined references: $(grep -c 'undefined' RND_JAX_exploration_development_document.log || true)"
/p/rlprojects/RND/.venvs/platform_jax/bin/python - <<'PY'
import fitz
doc = fitz.open('/p/rlprojects/RND/10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf')
out = '/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_uniform-decay-exponents-2-1-0.25_tabular-rnd-frozen-online_reward-normalization-off-on_steps-16777216_seeds-0-15_confirmation-20000-20127_baseline-to-21023/infrastructure/page_checks'
hits = [i for i, page in enumerate(doc) if 'tabular-RND' in page.get_text() or 'coordinate-RND' in page.get_text()]
print('pages', [i+1 for i in hits], 'of', len(doc))
for i in hits:
    doc[i].get_pixmap(dpi=110).save(f'{out}/page_{i+1}.png')
PY
chmod g+r $P/development_document/RND_JAX_exploration_development_document.pdf
