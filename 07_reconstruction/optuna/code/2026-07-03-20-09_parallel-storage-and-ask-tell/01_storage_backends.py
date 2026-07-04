"""Point 1 + point 6 docstring quotes: list the storage backends optuna 4.9.0 exposes,
quote the exact NFS / SQLite / parallel-optimization guidance from the docstrings, and show
which journal import paths are current versus deprecated (with the actual warning text)."""

import warnings
import tempfile
import os
import optuna


def list_storage_classes():
    """Print every public name under optuna.storages and optuna.storages.journal."""
    # top-level optuna.storages names
    print("=== public names under optuna.storages ===")
    for name in dir(optuna.storages):
        if not name.startswith("_"):
            print("  ", name)
    # the journal submodule (the current home of the file backend + lock objects)
    print("=== public names under optuna.storages.journal ===")
    for name in dir(optuna.storages.journal):
        if not name.startswith("_"):
            print("  ", name)


def quote_nfs_guidance():
    """Print the exact sentences the docstrings give about NFS, SQLite and parallel use."""
    # JournalFileBackend docstring: the sentence naming NFS and broken fcntl locking
    print("=== JournalFileBackend docstring (NFS / fcntl sentences) ===")
    print(optuna.storages.journal.JournalFileBackend.__doc__)
    # RDBStorage docstring ends with the explicit SQLite-for-parallel warning
    print("=== RDBStorage docstring: SQLite-for-parallel note (tail) ===")
    rdb_doc = optuna.storages.RDBStorage.__doc__
    # print only the note that names SQLite3 and parallel optimization
    idx = rdb_doc.index("We would never recommend SQLite3")
    print(rdb_doc[idx:idx + 160])


def quote_lock_guidance():
    """Print the docstrings that say which lock object goes with which NFS version."""
    # OpenLock: NFSv3 or later on kernel 2.6 or later
    print("=== JournalFileOpenLock docstring (which NFS) ===")
    print(optuna.storages.journal.JournalFileOpenLock.__doc__)
    # SymlinkLock: NFS prior to v3
    print("=== JournalFileSymlinkLock docstring (which NFS) ===")
    print(optuna.storages.journal.JournalFileSymlinkLock.__doc__)


def show_import_path_deprecations():
    """Instantiate the current and the deprecated journal names and print any warnings."""
    # make a throwaway journal file path
    directory = tempfile.mkdtemp()
    file_path = os.path.join(directory, "j.log")

    # current, non-deprecated path: optuna.storages.journal.JournalFileBackend -> no warning
    print("=== current: optuna.storages.journal.JournalFileBackend ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = optuna.storages.journal.JournalFileBackend(file_path)
        print("  type:", type(backend).__name__)
        print("  warnings:", [str(w.message) for w in caught] or "NONE")

    # deprecated path: optuna.storages.JournalFileStorage -> FutureWarning telling you the new name
    print("=== deprecated: optuna.storages.JournalFileStorage ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = optuna.storages.JournalFileStorage(os.path.join(directory, "j2.log"))
        print("  type:", type(old).__name__)
        for w in caught:
            print("  warning:", type(w.message).__name__, "->", w.message)

    # deprecated path: top-level optuna.storages.JournalFileOpenLock -> also warns
    print("=== deprecated: optuna.storages.JournalFileOpenLock (top-level) ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_lock = optuna.storages.JournalFileOpenLock(file_path)
        print("  type:", type(old_lock).__name__)
        for w in caught:
            print("  warning:", type(w.message).__name__, "->", w.message)


def quote_n_jobs_and_rdb_parallel():
    """Print the Study.optimize n_jobs docstring (threads) and the RDB parallel-alternative note."""
    # Study.optimize: the n_jobs argument description says it uses threads
    print("=== Study.optimize docstring: n_jobs paragraph ===")
    opt_doc = optuna.study.Study.optimize.__doc__
    idx = opt_doc.index("n_jobs:")
    print(opt_doc[idx:idx + 520])


# run every check in order
list_storage_classes()
quote_nfs_guidance()
quote_lock_guidance()
show_import_path_deprecations()
quote_n_jobs_and_rdb_parallel()
