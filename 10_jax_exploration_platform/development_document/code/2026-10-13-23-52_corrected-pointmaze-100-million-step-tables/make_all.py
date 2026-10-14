"""Regenerate the corrected results table and both training figures from the run aggregate."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from write_corrected_report import main

if __name__ == "__main__":
    # The shared writer preserves captions and prose; only the named table block is replaced.
    sys.argv = [sys.argv[0], "--run-dir", str(ROOT / "runs" / '2026-10-13-22-13_pointmaze_full-batch_steps-100-million_seeds-0-127_rollout-128_environments-4_random-network-distillation_weights-1-10-100-1000_visit-count-square-root_weights-1-10_visit-count-inverse_weight-1_no-bonus_learning-rates-1e-3-1e-4'),
                "--tex", str(ROOT / "development_document/RND_JAX_exploration_development_document.tex")]
    main()
