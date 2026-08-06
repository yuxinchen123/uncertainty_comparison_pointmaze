# Section 9 — structure, classification, and exact removal ranges

File: `/p/rlprojects/RND/07_reconstruction/development_document/RND_development_document.tex`
Section 9 spans **lines 7987–8639** (`\clearpage` before `\section` at 7988; `\bibliography{bibliography}` at 8641). Nothing was edited.

A mirror copy of this section exists at `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-08-05_cleanrl-rnd-writeup/section_draft.tex` (line 7977 says it is "kept in sync by hand"), with the same blocks at draft-file lines 9–611. Any cut applied to the main file must be mirrored there or the two will diverge.

---

## 1. Full structure, with classification

| Lines | Block | Covers | Class |
|---|---|---|---|
| 7975–7986 | Two duplicated `%` banner comments | Draft-source and figure-generator pointers. The banner is literally pasted twice (7975–7980 and 7981–7986). | infrastructure-flavored comment (invisible in the PDF; not part of the ask) |
| 7987–7989 | `\clearpage`, `\section{CleanRL PPO $+$ RND on Atari}`, `\label{sec:cleanrl}` | Section head | — |
| 7991–7996 | Opening paragraph | Why a second, independently written RND is worth checking against ours; names the file and commit `fe8d8a0` | **SCIENCE** |
| 7998–8003 | Second paragraph | Different domain / base learner / two value heads; CleanRL's 7100 is one seed vs Burda 8152 | **SCIENCE** |
| 8005–8006 | `\subsection{Train run 1}`, `\label{sec:cleanrl-trainrun1}` | — | — |
| 8008 | "This run asks four questions." | Lead-in to the enumerate | **SCIENCE** (needs a number edit — see §6) |
| 8010–8026 | `enumerate` of four questions | — | mixed |
| 8011–8013 | question 1 | Does the single-seed 7100 hold across 30 seeds | **SCIENCE** |
| 8014–8018 | question 2 | What the auto-reset defect costs | **SCIENCE** |
| 8019–8022 | question 3 | Where the two RND implementations differ | **SCIENCE** |
| **8023–8025** | **question 4** | **"What does a faithful reproduction of this workload cost on this cluster, and how fast can it be made to run?" — 250 hours, profiling** | **INFRASTRUCTURE** |
| 8031–8036 | `\paragraph{The two RND implementations, knob by knob.}` | Introduces the comparison table | **SCIENCE** |
| 8038–8128 | Table `tab:cleanrl-vs-project-rnd` | Knob-by-knob comparison of the two RND stacks (base learner, bonus input, networks, readout, intrinsic reward, run scale) | **SCIENCE** |
| 8130–8136 | Figure-intro paragraph | Introduces the pipeline figure; last sentence (8132–8136) is about print width of `fig:trainrun5-pipeline` and why its labels are hard to read | **SCIENCE**, with a document-typesetting tail (not cluster infrastructure — see §3) |
| 8138–8150 | Figure `fig:cleanrl-rnd-pipelines` | Two pipelines side by side | **SCIENCE** |
| 8155–8160 | `\paragraph{Environment specification.}` | Introduces the envpool table | **SCIENCE** |
| 8162–8229 | Table `tab:cleanrl-envspec` | Every envpool key in force: identity/spaces, preprocessing, episode, reward | **SCIENCE** (one infrastructure clause at 8180: "no download, so the environment builds on a node with no internet") |
| 8234–8239 | `\paragraph{Algorithms.}` | Introduces the hyperparameter table; promises "paragraphs that follow the table" justifying each code change | **SCIENCE** (needs rewrite — see §6) |
| 8241–8307 | Table `tab:cleanrl-trainrun1-algos` | RL algorithm block, intrinsic-bonus block, run-bookkeeping block, then the "changes to the code" block | mixed — see §3 |
| 8312–8313 | `\subsubsection{The environment auto-reset defect...}`, `\label{sec:cleanrl-bug}` | — | **SCIENCE** |
| 8315–8318 | Intro | CleanRL's own bug admonition; two of three claims need revising | **SCIENCE** |
| 8320–8343 | `\paragraph{What envpool actually does.}` | Next-step auto-reset, two measured probe transcripts, differential test | **SCIENCE** |
| 8345–8353 | `\paragraph{What that does to the rollout buffer.}` | Fabricated transition enters both value losses and the predictor loss; ~1.2% of rows | **SCIENCE** |
| 8355–8367 | `\paragraph{Why the damage is larger...}` | Non-episodic intrinsic return has no terminal mask, so the fabricated row leaks backwards at $0.94^{\Delta}$ | **SCIENCE** |
| 8369–8390 | `\paragraph{The correction.}` + 2-item enumerate | Carry advantage through the consumed row; drop consumed rows; why `elapsed_step` fails; why no upstream fix is coming; episode-length counter off by one | **SCIENCE** |
| 8395–8396 | `\subsubsection{Logging and resumability}`, `\label{sec:cleanrl-resume}` | — | **INFRASTRUCTURE** |
| 8398–8407 | `\paragraph{Logging.}` | No tensorboard/wandb; JSON record fields; temp file + atomic rename; `completed` flag | **INFRASTRUCTURE** |
| 8409–8413 | Episode-cap deviation | Keep every episode to 50,000, then every hundredth; store cap and stride | **INFRASTRUCTURE** with one science residue (see §5) |
| 8415–8436 | `\paragraph{Why resumability is required, and what is saved.}` + itemize | 4-day partition limit; the six things a checkpoint carries; 50.2 MB, 1.5 GB, 8-hour cadence | **INFRASTRUCTURE** |
| 8438–8443 | `\paragraph{What cannot be saved...}` | envpool emulators not serializable; a resumed run is not bit-identical | **INFRASTRUCTURE** with one science residue (see §5) |
| 8445–8450 | `\paragraph{Verified, not assumed.}` | The resume demonstration transcript | **INFRASTRUCTURE** |
| 8455–8456 | `\subsubsection{Throughput...}`, `\label{sec:cleanrl-throughput}` | — | **INFRASTRUCTURE** |
| 8458–8463 | Intro | 250 hours = 2,222 steps/s; 7,500 GPU-hours; pointer to the profiling folder | **INFRASTRUCTURE** |
| 8465–8491 | `\paragraph{The published run was starved of processor cores...}` + Table `tab:cleanrl-cores` | steps/s vs cores on one RTX A4500 | **INFRASTRUCTURE** |
| 8493–8495 | Eight-cores decision | 93% of peak on half the cores; 400/80 core caps | **INFRASTRUCTURE** |
| 8497–8515 | `\paragraph{Changes that leave the arithmetic unchanged...}` | Eight cumulative changes, 1.35–1.57× across seven GPU models; `uint8` buffer; start-up 4–7× faster | **INFRASTRUCTURE** with one science residue (see §5) |
| 8517–8544 | `\paragraph{A trap: the correction and the convolution autotuner...}` + Table `tab:cleanrl-shape` | Drifting minibatch shape re-tunes the autotuner every update; fixed row allowance recovers it | **INFRASTRUCTURE** with one science residue (see §5) |
| 8546–8578 | `\paragraph{Several runs fit on one GPU...}` + Table `tab:cleanrl-packing` | Packing N runs per GPU; which pool is core-scarce vs GPU-scarce | **INFRASTRUCTURE** |
| 8580–8584 | `\paragraph{What the run costs.}` | 5–9 days per seed, ~3,900 GPU-hours, wall-clock, one resume per seed | **INFRASTRUCTURE** |
| 8589–8607 | `\paragraph{Provenance.}` | Upstream commit and untouched clone (science); run-folder path, venv path, profiling path (infrastructure) | **MIXED** — trim, do not delete |
| 8609–8618 | `\paragraph{Status at the 2026-08-05 snapshot.}` | Canary, node-by-node slot assignment, GPU models, spare slots, days per seed; last sentence says no outcome is reported yet | **INFRASTRUCTURE** except the final sentence |
| 8620–8636 | `\paragraph{Two things the canary caught that review had not.}` + 2-item enumerate | (1) one-row minibatch → NaN; (2) Slurm cores-per-task reported per job, not per run | **MIXED** — see §3 |

---

## 2. The borderline cases, argued

**The auto-reset defect subsubsection (8312–8390) — does any part drift into infrastructure? No.**
I checked every paragraph for the infrastructure markers and found none: no cluster, no node, no steps/s, no checkpoint, no file format, no job. The three things that could be mistaken for infrastructure are all correctness:
1. The probe transcripts at 8328–8339 and the differential test at 8341–8343 are *evidence for a semantic claim about the environment*, not a measurement of the machine. Delete them and the claim "envpool consumes the following step" becomes an assertion.
2. The NaN mechanism in correction item 2 (8378–8382) is numerics, not plumbing: a zero-standard-deviation divide reaching the weights changes the model, not the throughput.
3. The episode-length counter at 8388–8390 sounds like logging, but the defect is that a *reported quantity* (episode length) is wrong by one per life lost. A wrong metric is a wrong result, not a logging mechanism.
   Keep 8312–8390 intact.

**The logging-and-resumability subsubsection (8395–8450) — all infrastructure? Effectively yes, with two residues.**
Four of the five paragraphs are pure mechanism: file format and atomic rename (8398–8407), what a checkpoint carries and how big it is (8415–8436), and the resume demonstration (8445–8450). Two facts are not mechanism:
- **8409–8413**: the record keeps every episode up to 50,000 and every hundredth after. That is a statement about *what data exist for analysis* — a reader of a future results table needs to know the episode series is subsampled beyond 50,000. It is stated here as a file-size decision but it constrains the science.
- **8438–8443**: envpool state cannot be serialized, so a resumed run restarts its 128 environments and is not bit-identical to an uninterrupted one. That is a reproducibility caveat on the run's own data.
  Everything else in this subsubsection goes, including the frozen-target checkpoint bullet at 8424–8426 — the *reason* it matters (a randomly initialized target defines the bonus) is already stated in `tab:cleanrl-vs-project-rnd` and in the pipeline figure.

**The throughput subsubsection (8455–8584) — all infrastructure? Yes, with two residues that must be relocated.**
Every table and every number in it is cluster measurement. Two sentences carry science:
- **8510–8511**: "The networks divide by $255$ either way and every `uint8` value is exactly representable in `float32`, so the arithmetic is untouched." This is the guarantee that the speed work did not change the computation — i.e. that this run is still comparable to CleanRL's published number. Without it, a reader has no basis to trust the 30-seed distribution against the published 7100.
- **8538–8542**: the fix for the autotuner interaction is "reserving a fixed row allowance", and the caption notes "the constant-shape batch is marginally smaller". That is a change to *what the optimizer sees* — a slightly smaller and constant-size update batch. It belongs in the correction subsubsection, not in a throughput table. If it is deleted with the rest, the section silently loses a description of the batch the corrected code actually trains on.

**"Status at the snapshot" (8609–8618) — infrastructure? Yes, except the last sentence.**
Lines 8610–8616 are canary, worker slots, six named nodes with their GPU models, spare slots, and days per seed — every one of these is on the infrastructure list. Lines 8617–8618 ("nothing in this section reports an outcome yet") are a statement about the state of the *results*, which a reader needs, and which the owner did not ask to remove.

**"Two things the canary caught" (8620–8636) — mixed; delete the whole paragraph.**
Item 2 (8630–8635) is pure infrastructure: Slurm's cores-per-task variable, packed nodes, oversubscription, and it says outright "This one changes no result; it only costs throughput."
Item 1 (8624–8629) is *not* infrastructure — a NaN in the weights is a correctness defect. But it is **already stated**, in the same detail, in correction item 2 at 8378–8382 ("a ragged final slice of one row makes the advantage normalization divide by a zero standard deviation, and the resulting NaN reaches the weights within one optimizer step. The batch is then walked as exactly `num_minibatches` equal slices with the remainder dropped"). So item 1 is deleted as a duplicate, not as infrastructure. The only fact here that is not already upstream is "the shuffle means a different remainder each epoch" (8629) — fold it into 8381 if you want to keep it. With both items gone, the lead-in at 8620–8621 has nothing to lead into, so the paragraph goes as a unit. Note the framing itself ("the canary caught") is a job-submission concept.

**The hyperparameter table's last block (8288–8297) — row by row:**

| Lines | Row | Class |
|---|---|---|
| 8289 | Block header "Changes this run makes to the code, not to the algorithm" | keep, retitle to singular |
| 8290–8291 | `environment auto-reset defect & corrected` → `\Cref{sec:cleanrl-bug}`; "the only change that alters what the optimizer sees" | **SCIENCE** — keep |
| 8292–8293 | `metric logging` — one JSON per run, no wandb, no tensorboard | **INFRASTRUCTURE** — remove |
| 8294–8295 | `checkpoint and resume` — every 8 hours, previous replaced, 4-day partition limit | **INFRASTRUCTURE** — remove |
| 8296–8297 | `throughput options` — "the options that leave the computation unchanged are on; the options that change it are off" | **INFRASTRUCTURE** in form, but this row is the *only* place in the whole table asserting that the speed work does not alter the arithmetic. Remove the row, keep the assertion as prose (see §5). |

---

## 3. Exact removal ranges

Ordered so that the line numbers below are all read against the **current, unedited** file. Apply from the bottom up so earlier ranges do not shift.

| # | Range (inclusive) | What it is |
|---|---|---|
| R1 | **8023–8025** | Question 4 of the four-question enumerate (cost and speed). Line 8022 ends question 3; 8026 is `\end{enumerate}`. |
| R2 | **8292–8297** | Three rows of `tab:cleanrl-trainrun1-algos`: metric logging, checkpoint and resume, throughput options. Line 8291 ends the auto-reset row; 8298 is `\bottomrule`. |
| R3 | **8392–8451** | `% ---` banner (8392–8394), `\subsubsection{Logging and resumability}` and all five of its paragraphs, plus the trailing blank line 8451. |
| R4 | **8452–8585** | `% ---` banner (8452–8454), `\subsubsection{Throughput...}`, its four paragraphs and three tables, `\paragraph{What the run costs.}`, plus the trailing blank line 8585. |
| R3+R4 | **8392–8585** | The two above are contiguous and can be taken as one clean cut. Line 8391 is blank (end of `sec:cleanrl-bug`); line 8586 opens the Provenance banner. |
| R5 | **8610–8616** | Body of `\paragraph{Status at the 2026-08-05 snapshot.}` — canary, 30 worker slots, the six named nodes and their GPU models, spare `gnolim` slots, days per seed. Keeps the `\paragraph` line 8609 (retitle) and the closing sentence 8617–8618. |
| R6 | **8620–8636** | The whole `\paragraph{Two things the canary caught that review had not.}` block including its `enumerate`. Line 8619 is blank; 8637–8638 blank; 8639 `\clearpage`. |
| R7 (optional, trim not delete) | **8593–8600** and **8604–8607** | Inside `\paragraph{Provenance.}`: the `samepage` quote holding the full run-folder slug, and the venv + profiling paths. See §5 — the profiling path is the one thing worth keeping, as the replacement pointer. |
| R8 (optional) | **8132–8136** (from "The figure is drawn vertically" to "hard to read.") | Document-typesetting trivia about print widths. Not cluster infrastructure, so not covered by the owner's instruction; flagged only because it is the one remaining passage that talks about mechanics rather than content. If cut, see §4 for the `fig:trainrun5-pipeline` reference. |
| R9 (optional) | **8180**, the clause "no download, so the environment builds on a node with no internet" | A single infrastructure clause inside a science table cell. |

Total removed by R1–R6: **214 lines**.

---

## 4. Cross-reference analysis

### Labels defined inside the removed ranges

| Label | Defined at | Inside range | Referenced at | Survives the cut? |
|---|---|---|---|---|
| `sec:cleanrl-resume` | 8396 | R3 | **8295** only (the "checkpoint and resume" table row) | Yes — 8295 is removed by R2. Zero dangling. |
| `sec:cleanrl-throughput` | 8456 | R4 | **8025** (question 4), **8297** (throughput-options row), **8417** (inside the resumability paragraph) | Yes — 8025 removed by R1, 8297 by R2, 8417 by R3. Zero dangling. |
| `tab:cleanrl-cores` | 8490 | R4 | nowhere | Yes — the surrounding prose introduces the table with a colon, not a `\Cref`. |
| `tab:cleanrl-shape` | 8543 | R4 | nowhere | Yes — same. |
| `tab:cleanrl-packing` | 8571 | R4 | nowhere | Yes — same. |

**No label defined outside section 9 is removed, and no label defined in section 9 is referenced from outside section 9.** A whole-file grep for `sec:cleanrl`, `tab:cleanrl`, `fig:cleanrl` returns 26 hits, all between lines 7989 and 8591 — the section is referentially self-contained. `sec:cleanrl` itself (7989) has zero references anywhere in the document.

**Surviving labels in section 9 after the cut**: `sec:cleanrl` (7989), `sec:cleanrl-trainrun1` (8006), `tab:cleanrl-vs-project-rnd` (8127), `fig:cleanrl-rnd-pipelines` (8149), `tab:cleanrl-envspec` (8228), `tab:cleanrl-trainrun1-algos` (8306), `sec:cleanrl-bug` (8313). All still referenced from within the section, so `cleveref` stays quiet.

### References *out* of section 9 that the removal destroys

These point at labels defined earlier in the document. The labels survive; only the citing text disappears.

| Target label | Defined at | Cited from removed lines | Effect |
|---|---|---|---|
| `sec:logging` | 2001 | **8293** (R2), **8400** and **8409** (R3) | After the cut, `sec:logging` has **zero references in the entire document**. That is not an error — LaTeX and `cleveref` never warn about an unreferenced label — but section 9 will no longer connect to the project's logging convention at all. Deliberate, given the instruction. |
| `fig:trainrun5-pipeline` | 6089 | 8134 — only if optional R8 is taken | Still referenced at 6070 (`Figure~\ref{...}`), so no orphan either way. |

Untouched and still cited from surviving section-9 prose: `tab:pm-am-trainrun1-algos` (7192, cited at 8034/8052/8236/8302), `tab:trainrun5-hparams` (6233, cited at 8035/8121), `sec:pm-am` (6826, cited at 7992/8034/8114/8303), `tab:pm-am-envspec` (6958, cited at 8220).

### Non-label pointer that breaks

Line **8461–8463** (inside R4) is the only place in the `.tex` naming the profiling folder `08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling/` other than 8605–8607 in Provenance. If both R4 and the R7 trim are taken, that path leaves the document entirely — which is why the replacement sentence in §5 keeps it.

---

## 5. Replacement text

Five short insertions. Together they are 6 sentences replacing 214 lines.

1. **For R4 (throughput), one sentence, placed at the end of `\paragraph{Provenance.}` around line 8607.** This is also the fix for the lost profiling pointer:

   > A separate profiling study measured this workload's throughput on this cluster and set the run's processor allocation, its GPU packing, and the speed changes it applies; it is recorded in `08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling/` and none of its changes alter the arithmetic the optimizer sees.

   The trailing clause carries the guarantee that the deleted table row 8296–8297 and the deleted sentence at 8510–8511 were making.

2. **For the science residue at 8538–8542 (constant-shape batch), one sentence appended to correction item 2, after line 8382**, so the description of what the corrected code actually trains on survives:

   > The surviving-row count is then held at a fixed allowance rather than varying with the number of rows dropped, so every update trains on an equal-sized batch.

3. **For R3 (logging and resumability), at most one sentence** — and it is optional. If the owner wants the two data-facing residues kept, put them in `\paragraph{Provenance.}`:

   > The run writes one JSON record per seed and checkpoints every eight hours; the record keeps every episode up to $50{,}000$ and every hundredth after that, and a resumed run restarts its environment copies, so it is not bit-identical to an uninterrupted one.

   If the owner prefers a hard cut, drop this — nothing else in the section depends on it today, though a future results subsubsection reading the episode series will need the subsampling stated somewhere.

4. **For R1 (question 4)**, no replacement. Delete the item and change line 8008 from "This run asks four questions." to "This run asks three questions."

5. **For R5 (status)**, retitle line 8609 from `\paragraph{Status at the 2026-08-05 snapshot.}` to `\paragraph{Status.}` and keep 8617–8618, with "this subsubsection" corrected to "this section" (it was already wrong — the sentence sits at subsection level, after the last subsubsection):

   > All $30$ seeds were launched on 2026-08-05 and are still training. Results tables and reward curves will be added to this section when the seeds finish; nothing in this section reports an outcome yet.

---

## 6. Orphaned prose left by the removals

Six places, all inside surviving text, all of which must be edited or they become false or dangling:

1. **Line 8008 — "This run asks four questions."** R1 leaves three. Change the numeral.
2. **Lines 8236–8239 — the `\paragraph{Algorithms.}` lead-in**: "What the middle column records instead is the small number of things this run changes about the *code* rather than the algorithm, **each of which is justified in the paragraphs that follow the table**." After R2 the block holds one row, and after R3+R4 the justifying paragraphs for logging, checkpointing and throughput no longer exist. Rewrite to something like: "…the one thing this run changes about the *code* rather than the algorithm, which \Cref{sec:cleanrl-bug} justifies."
3. **Lines 8303–8305 — the table caption**: "The last block lists the change**s** this run makes to the *code*; only the **first of them** changes what the optimizer sees." Both the plural and "first of them" are wrong once only one row remains. Rewrite: "The last block lists the one change this run makes to the *code*, which is also the only change to what the optimizer sees."
4. **Line 8289 — the block header** `\textit{Changes this run makes to the code, not to the algorithm}`. Singular after R2: "Change this run makes to the code, not to the algorithm".
5. **Line 8591 — Provenance**: "the run executes a copy with **the four changes** of `\Cref{tab:cleanrl-trainrun1-algos}`." R2 leaves one. Change to "the one change of".
6. **Lines 8620–8621** — the lead-in "Both are recorded because they are the kind of defect that does not announce itself" is orphaned by R6; it is inside the R6 range, so it goes with it. Listed here only so it is not accidentally kept.

Two checks that came back clean:
- **No surviving sentence introduces a deleted table.** `tab:cleanrl-cores`, `tab:cleanrl-shape` and `tab:cleanrl-packing` are each introduced by a colon-ended sentence that lives inside R4 and is deleted with them.
- **No "as measured above" / "as noted above" / "shown above" / "earlier in this section" anywhere in lines 7988–8640.** The only backward pointers are explicit `\Cref`s, all accounted for in §4.