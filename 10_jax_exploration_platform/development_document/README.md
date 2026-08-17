# development_document

The platform's writeup: the LaTeX document, its style file, its bibliography, and the table
generators.

| file / folder | what it is |
|---|---|
| `platform_development_document.tex` | the document itself |
| `neurips_2026.sty` | the paper style, copied from `07_reconstruction/development_document/` |
| `bibliography.bib` | the bibliography, copied from the same place |
| `code/` | one folder per run's table generator, named `<YYYY-MM-DD-HH-MM>_<run-slug>-tables/` |

## Building it

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -cd -pdf -interaction=nonstopmode \
  /p/rlprojects/RND/10_jax_exploration_platform/development_document/platform_development_document.tex
```

The built PDF is gitignored and is never committed.

## What a train run adds

The shared skill `/p/rlprojects/.claude/skills/rnd-experiment-tex-track/SKILL.md` owns the shape of
every run's block: an environment-specification table (only for an environment the document does
not yet specify), a what-is-swept table, a best-configuration results table, and a training-curve
figure. The first two are written by hand at launch; the last two are written by the run's
generator under `code/` and spliced between the `% >>> AUTO-GENERATED TABLE START: <name>` /
`% <<< AUTO-GENERATED TABLE END: <name>` markers in the `.tex`.

## Numbering

`\section{PointMaze on the JAX platform}` opens with `\setcounter{subsection}{-1}`, so the shared
environment specification is subsection 1.0 and the first train run is 1.1. A later run on the same
environment becomes 1.2 and points back at the 1.0 table instead of repeating it.

## After every edit or regeneration

1. Rebuild the PDF in the same turn.
2. `grep -nE 'Overfull \\hbox' platform_development_document.log` — a hit at a table's lines means
   it overflows; fix it by folding the cell text (`\allowbreak`, `\shortstack`, rebalanced column
   widths), never by shrinking the font, and never by rotating a column header.
3. Rasterize the changed pages (`gs -sDEVICE=png16m -r150 -dFirstPage=N -dLastPage=N`) and look at
   them: the grey verticals continuous through every black rule, every spanned cell bounded on both
   sides, nothing crossing a column edge.
4. Commit the `.tex` and the generator folder, never the PDF.
