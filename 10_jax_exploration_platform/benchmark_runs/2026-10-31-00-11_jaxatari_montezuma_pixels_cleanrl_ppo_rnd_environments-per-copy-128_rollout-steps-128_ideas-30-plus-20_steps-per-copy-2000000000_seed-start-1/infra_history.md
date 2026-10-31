# Infrastructure history

Hand-written session record. Times are Pacific.

- October 31, 2026, 00:11: created integration worktree; original checkout local changes preserved.
- Source reference checkout and isolated environment preparation started before any training or benchmark.
- ACES and DeltaAI existing SSH connections were verified from portal05. No jobs submitted there yet.

- First pixel jobs 6777084 and 6777085 failed before Python started because new NFS snapshot files were not visible on the compute host. Snapshot publication is now atomic and job scripts check host-side visibility before sourcing files. Failed job IDs remain in the ledger; replacement jobs are recorded separately.

- Pixel baseline numerical snapshot: 21 checks passed; one expected-array fixture was read-only and is corrected. Expanded snapshot: 23 passed; two custom kernel tests exposed pinned JAX Pallas API and RTX A4500 target-name metadata omissions, being corrected before acceptance.
- Small real-game checks passed on jaguar03 and DeltaAI; serval02 card1 small check also completed. These tiny checks are not representative throughput measurements.

## 2026-10-31 01:43 PT: deterministic checks and renderer constructor

The full A4500 learner suite passed 46 checks in job 6777336. Enabling deterministic XLA operations resolved exact-copy and checkpoint-continuation differences without changing test tolerances. Configuration-file job 6777337 passed. Both renderer checks found the optional upstream background-cache template missing prev_is_fire; the adapter now builds static room backgrounds from a complete reset state. The H100 learner check did not collect tests because its launcher omitted the source import path; the launcher now sets it explicitly. All failures remain in their original output records. Formal paired comparisons have not begun.

Preliminary one-copy resource probes completed at 9,114 steps per second on serval02 H100 and 3,712 steps per second on one jaguar03 RTX A4500. These used the earlier runtime configuration and are not accepted optimization comparisons. Two-copy probes will test memory and packing before choosing the common comparison grid.

## 2026-10-31 02:04 PT: frozen source checks and remote reference probe

The current RND branch has been fast-forwarded to 48138c2 and pushed. All five pre-existing local files were checked before and after and remained byte for byte identical. The compiled report now has 39 pages. Source d28e673 passed 48 learner tests and all three sets of 132 rendering cases plus 11 synthetic pixel fixtures on jaguar03. The earlier 1d488ab learner suite passed all 47 tests on serval02. Updated H100 checks wait behind its full-size two-copy resource probe under the shared card-1 lock. The A4500 two-copy probe is also still measuring. No paired optimization case has been admitted.

ACES reference resource probe 2271806 is submitted on the normal account and was pending for priority at submission. Its exact identifier is recorded locally and remotely; its source archive hash and paths are under remote_setup/aces_reference_probe. The persistent UVA monitor is 6777344, and recurring session task 72a562a5-596b-41c0-af66-98beb4034a8e remains every twenty minutes. Only our superseded monitor 6777069 was stopped after the replacement was verified.

The reviewed admission helper now requires actual completed capacity evidence for each requested copy count above one on both GPU types, exact complete check commands from the frozen source, and immutable evidence records. next_actions.json records the concrete guarded admission and worker commands. Before the later adaptive comparisons, reference environment settings must be included in immutable case metadata and forwarded into the paired benchmark so that the accepted environment changes become its baseline. The initial comparisons use the original reference environment.

## 2026-10-31 02:36 PT: H100 execution investigation

The deterministic two-copy H100 probe is progressing, but its first measured rounds take about 451 seconds per ten updates, around 727 transitions per second in total. Its initialization took 317 seconds; this is not an initialization delay. The selected card has only the recorded process, full GPU use, power near its 400-watt limit, and no thermal slowdown. The A4500 two-copy probe completed seven rounds, around 2,985 transitions per second. The deterministic one-copy ACES probe is also progressing at about 821 transitions per second on its H100 PCIe. These are resource observations, not accepted optimization comparisons.

Copy count, runtime flags, GPU model, and convolution lowering differ across these observations. JAX maps independent inputs and per-policy kernels to grouped convolutions. Pinned XLA documentation states deterministic scatter, matrix-product, and convolution paths can be slower; its cuDNN code filters nondeterministic plans and stops at the first working deterministic plan. This identifies possible causes, without attributing the observed cost to a particular kernel. Source links: https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/docs/determinism.md and https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/stream_executor/cuda/cuda_dnn.cc#L5418 .

A separate full-size one-copy diagnostic will retain deterministic operations and all original precision settings. It waits for the current probe and the updated H100 checks, then measures four updates after two compile updates and traces one later update. It records actual loaded CUDA and cuDNN libraries and kernel durations. No formal result is admitted from this diagnostic. The existing measurements and queued checks remain unchanged.

The future adaptive case metadata now supports accepted environment performance settings as its reference baseline. Protocol changes remain forbidden, and empty settings preserve the initial case contents. Six focused metadata tests and the existing twenty-seven queue tests passed. No additional research hypotheses have been chosen.

## 2026-10-31 03:39 PT: verified convolution cost and matched controls

All four source-d28 learner and renderer checks passed on both GPU types, including all 48 learner tests and the 132 rendering states for each of three variants plus 11 synthetic pixel cases. Both full-size two-copy capacity records completed seven rounds and passed the metadata admission checks. Source admission remains unpublished while the H100 execution diagnosis finishes.

The one-copy deterministic H100 diagnostic completed. Its two stored timing rounds took 13.139 and 13.155 seconds per update. Actual loaded libraries are CUDA 12.9 and cuDNN 9.24. Nsight created a raw capture, but SQLite summary export at the long output path failed; Nsight nevertheless returned exit zero. The original helper incorrectly reported summary completion. The original outputs were preserved. CPU-only recovery through a short local directory produced valid summaries in trace/recovery_2026-10-31-03-22-29: convolution weight-gradient kernels account for 86.4 percent of summed GPU kernel time, across 32 calls. The recovered files include 442 kernel rows and 21 API rows, and the input report hash was verified.

New diagnostic helpers export summaries through a local temporary path and require actual positive kernel and API data before reporting completion. Cache inputs are isolated and recorded. Matched controls compare the original setting with disabled determinism on the same H100 and A4500. A separate deterministic H100 control forces the supported channels-last convolution layout while keeping the original logical model and learning configuration. Every control retains the same frozen core and seven-update diagnostic protocol. Disabled-determinism controls cannot become formal benchmark or capacity evidence. All 15 affected metadata tests passed; the earlier full campaign metadata suite passed 55 tests. No learning hyperparameter search or additional research hypothesis selection has occurred.
