"""Options of report_job.py, which runs inside the test helper's Slurm job (pytest is what that job runs)."""


def pytest_addoption(parser):
    """Let the command line name the run folder and the report's options."""
    parser.addoption('--run-dir', default=None,
                     help='run folder whose report is generated (relative to the platform folder, or absolute)')
    parser.addoption('--at-steps', type=int, default=None,
                     help='compare every table at this many environment steps per policy')
    parser.addoption('--latex-check-dir', default=None,
                     help='also compile results.tex in this folder (on shared storage) and write its pages as images')
