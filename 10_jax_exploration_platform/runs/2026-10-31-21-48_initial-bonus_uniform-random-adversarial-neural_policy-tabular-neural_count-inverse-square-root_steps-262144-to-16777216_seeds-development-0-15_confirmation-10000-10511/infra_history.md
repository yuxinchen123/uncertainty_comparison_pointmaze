# Infrastructure history

Initial verification stopped before numerical work because platform_jax has no pytest. The reference environment already includes pytest with the same JAX versions. Verification now runs tests there and separately exercises the production GPU runtime. Source revisions are recorded in source.json and job submissions.

Primary verification completed with 84 passing tests and both production record checks. Compute nodes had neither Poppler nor Ghostscript; PyMuPDF 1.26.6 now renders the compiled tables. Twenty initial performance ideas finished: nineteen accepted and one rejected by the unchanged numerical tolerance, plus a complete baseline. Ten measured follow-up combinations are queued/running on all eight jaguar03 RTX A4500 cards. Per-task hardware records and logs are intact; the job-level hardware and assignment files currently retain the last GPU writer.
