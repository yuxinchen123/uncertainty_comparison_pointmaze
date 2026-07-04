"""Point 6: confirm the three one-sentence facts with exact docstring quotes and a functional check.
(a) study.optimize(n_jobs>1) uses THREADS in one process (so it is not the way to parallelize
CPU-heavy training); (b) separate worker processes sharing one storage is the way (see script 02);
(c) RDBStorage with a real database server (MySQL/PostgreSQL) is the documented alternative for
large parallel sweeps, but it needs a database server this cluster does not provide, whereas the
journal file backend needs none (it is just a file path)."""

import os
import tempfile
import optuna


def quote_n_jobs_threads():
    """Quote the Study.optimize n_jobs docstring lines that say it uses threading."""
    # pull the n_jobs paragraph straight from the live docstring
    doc = optuna.study.Study.optimize.__doc__
    start = doc.index("n_jobs:")
    end = doc.index("catch:")
    print("=== (a) Study.optimize n_jobs docstring ===")
    print(doc[start:end].rstrip())


def quote_rdb_parallel_alternative():
    """Quote the RDBStorage docstring lines about SQLite-for-parallel and the MySQL server option."""
    doc = optuna.storages.RDBStorage.__doc__
    # the explicit no-SQLite-for-parallel note
    i = doc.index("We would never recommend SQLite3")
    print("=== (c) RDBStorage: SQLite not for parallel ===")
    print(doc[i:i + 110].rstrip())
    # the MySQL server note (shows a real database server is the intended parallel backend)
    j = doc.index("If you use MySQL")
    print("=== (c) RDBStorage: MySQL server note ===")
    print(doc[j:j + 210].rstrip())


def rdb_needs_a_url_journal_needs_only_a_path():
    """Show RDBStorage is created from a database URL while JournalStorage needs only a file path."""
    directory = tempfile.mkdtemp()

    # RDBStorage is instantiated from a database URL (here in-process SQLite; a real sweep would
    # point at mysql://... or postgresql://..., i.e. a server that must be running separately)
    rdb = optuna.storages.RDBStorage(url="sqlite:///" + os.path.join(directory, "study.db"))
    rdb_study = optuna.create_study(storage=rdb)
    rdb_study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(-1, 1)},
            value=0.5,
        )
    )
    print("=== (c) RDBStorage functional (SQLite url), trials:", len(rdb_study.trials), "===")

    # JournalStorage needs only a file path plus a lock object; there is no server process
    journal_path = os.path.join(directory, "j.log")
    journal = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            journal_path, lock_obj=optuna.storages.journal.JournalFileOpenLock(journal_path)
        )
    )
    journal_study = optuna.create_study(storage=journal)
    print("=== (c) JournalStorage functional (file only, no server), trials:",
          len(journal_study.trials), "===")
    print("     journal file exists on disk:", os.path.exists(journal_path))
    print("(b) is verified separately by script 02: 8 worker PROCESSES sharing one journal study.")


# run the three checks
quote_n_jobs_threads()
quote_rdb_parallel_alternative()
rdb_needs_a_url_journal_needs_only_a_path()
