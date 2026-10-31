# Pixel Montezuma infrastructure report

`report_source.py` is the reviewed specification and source record. It names the
absolute campaign directory, the original CleanRL and JAXtari commits, and the
environment, learner, and timing settings. It contains no measured performance.

`make_all.py` reads `study_registry.json`, `campaign_state.json`, and only the
result files explicitly indexed by the registry. It checks the reviewed learner
and timing settings against `config_resolved.yaml`. Timing uses seven paired
rounds with at least one complete update per variant per round, five for grouped
updates, and ten for report accumulation. The selected combination requires a
separate seven-round confirmation with ten complete updates per variant per
round. It never searches resource-probe
directories for results. It uses the campaign's own investigation and paired
decision functions, and does not select a final implementation or start jobs.

The generator updates only the tabular blocks for these document labels:

- `tab:pixel-montezuma-environment`
- `tab:pixel-montezuma-learner`
- `tab:pixel-montezuma-networks`
- `tab:pixel-montezuma-study-status`
- `tab:pixel-montezuma-case-outcomes`

The document's captions, labels, and surrounding prose remain manual. Update
the current-results paragraph when the campaign enters a new stage; the
generator does not silently change scientific claims in prose.

From this folder, format the initial specifications and metadata with:

```bash
python3 make_all.py
```

Once a registered result exists, run the numerical decision audit on a compute
host before updating the document. Plotting also requires a compute host:

```bash
python3 make_all.py --analyze --plots
```

The allowed numerical-analysis contexts are a Slurm compute allocation or
serval02, serval05, or jaguar03. This command does not reserve a GPU or acquire a
machine lock; the campaign owner coordinates compute access. No training is run
by this generator. Matplotlib is needed only with `--plots`.

`data/registered_measurements.json` is the complete compute-audited aggregate.
It retains all indexed attempts, including rejected and inconclusive results.
Its input fingerprint covers the registry, every indexed result's bytes, and
the reviewed specification. A changed input prevents stale statistical output
from being used. Resource probes remain excluded even if they contain timing
fields. `analysis.md` is generated; do not edit it directly.

Plots separate GPU model, host, policy-copy count, source commit, reference
configuration, and measurement scope. Every valid paired ratio is shown, with
its median. Correctness rejections and invalid measurements remain in the
complete report rather than being plotted as zero speed. Report-readout timing
is separate from training-only timing. `plots/plot_manifest.json` records exact
plot inputs; old plot files are not automatically inserted into the paper.

After generation, build the document from the repository root:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.tex
```

Check the build log for layout warnings and inspect the rendered table pages.
No final training results or learning curves are produced before their actual
registered records exist.
