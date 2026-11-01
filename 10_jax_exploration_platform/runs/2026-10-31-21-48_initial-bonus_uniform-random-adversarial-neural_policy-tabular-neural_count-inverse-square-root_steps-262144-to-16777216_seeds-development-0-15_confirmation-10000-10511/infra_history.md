# Infrastructure history

Initial verification stopped before numerical work because platform_jax has no pytest. The reference environment already includes pytest with the same JAX versions. Verification now runs tests there and separately exercises the production GPU runtime. Source revisions are recorded in source.json and job submissions.

Primary verification completed with 84 passing tests and both production record checks. Compute nodes had neither Poppler nor Ghostscript; PyMuPDF 1.26.6 now renders the compiled tables. Twenty initial performance ideas finished: nineteen accepted and one rejected by the unchanged numerical tolerance, plus a complete baseline. Ten measured follow-up combinations are queued/running on all eight jaguar03 RTX A4500 cards. Per-task hardware records and logs are intact; the job-level hardware and assignment files currently retain the last GPU writer.

Heavy plot/PDF generation is now a separate owner CPU job, coalesced by completed-result count. The persistent control tick remains lightweight and keeps a twenty-minute cadence. Forty-three report, scheduler, and selection tests passed before this update. Selection minimizes capped first-goal time; discovery fraction breaks ties. GPU trainers continue from their original immutable snapshots because numerical trainer inputs did not change.

Large parameter lists are divided into independent units of at most sixteen settings, preserving every seed and parameter value. Selection pools those physical slices before ranking conditions. Forty-seven scheduler, reporting, pooling and queue-splitting checks passed. Numerical training inputs are unchanged.
