# Campaign tables and figures generator

`make_all.py` splices the ledger table between the `AUTO-GENERATED TABLE` markers of
`11_decay_rate_development_document.tex` and renders the three campaign figures into
`../../figures/`. Sources: the tracked ledger `11_decay_rate/results.tsv` and the campaign
folder's per-cell JSON records, read through the fixed harness modules (`decay_harness`)
and the diagnostics tool (`diagnose.py`) — nothing is re-implemented here, so the document
cannot drift from the measured records.

Run: `/p/rlprojects/RND/.venvs/exploration/bin/python make_all.py`
