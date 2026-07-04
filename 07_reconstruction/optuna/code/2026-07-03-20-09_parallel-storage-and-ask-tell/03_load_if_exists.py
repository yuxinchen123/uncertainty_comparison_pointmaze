"""Point 3: create_study(..., load_if_exists=True) attaches to an existing study instead of
raising; without load_if_exists a second create_study on the same name raises. Show the exact
exception type. Uses a journal file on NFS so the behavior matches the real sweep setup."""

import os
import optuna


def build_storage(journal_path):
    """Build the JournalStorage backed by a journal file with the NFSv3-recommended open lock."""
    # same storage construction every caller uses
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    )


def main():
    """First create the study; then a second create with and without load_if_exists."""
    # clean journal file so the study name starts unused
    here = os.path.dirname(os.path.abspath(__file__))
    journal_path = os.path.join(here, "03_study_journal.log")
    if os.path.exists(journal_path):
        os.remove(journal_path)
    storage = build_storage(journal_path)
    name = "point3_study"

    # first create: succeeds and records one trial so we can prove re-attachment sees it
    study_a = optuna.create_study(study_name=name, storage=storage, direction="minimize")
    study_a.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(-10, 10)},
            value=42.0,
        )
    )
    print(f"first create_study OK; study has {len(study_a.trials)} trial(s)")

    # second create WITH load_if_exists=True: attaches to the same study, sees the same trial
    study_b = optuna.create_study(
        study_name=name, storage=storage, direction="minimize", load_if_exists=True
    )
    print(f"load_if_exists=True re-attached; study_b has {len(study_b.trials)} trial(s)")
    print(f"  same study_id? {study_a._study_id == study_b._study_id}")
    print(f"  study_b sees first trial value: {study_b.trials[0].value}")

    # second create WITHOUT load_if_exists (default False): must raise; catch the exact type
    print("second create_study without load_if_exists:")
    try:
        optuna.create_study(study_name=name, storage=storage, direction="minimize")
        print("  ERROR: expected an exception but none was raised")
    except optuna.exceptions.DuplicatedStudyError as exc:
        # this is the specific exception the docs promise for a duplicate study name
        print(f"  raised {type(exc).__module__}.{type(exc).__name__}: {exc}")


main()
