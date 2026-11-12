"""Options of report_job.py and batch_summary_job.py, which run inside the test helper's Slurm job (pytest is what that job runs)."""


def pytest_addoption(parser):
    """Let the command line name the run folder and the report's options."""
    parser.addoption('--run-dir', default=None,
                     help='run folder whose report is generated (relative to the platform folder, or absolute)')
    parser.addoption('--batch', default=None,
                     help="batch_summary_job.py: the batch whose summary is generated, a folder name under the run folder's batches/")
    parser.addoption('--at-steps', type=int, default=None,
                     help='compare every table at this many environment steps per policy')
    parser.addoption('--latex-check-dir', default=None,
                     help='also compile results.tex in this folder (on shared storage) and write its pages as images')
