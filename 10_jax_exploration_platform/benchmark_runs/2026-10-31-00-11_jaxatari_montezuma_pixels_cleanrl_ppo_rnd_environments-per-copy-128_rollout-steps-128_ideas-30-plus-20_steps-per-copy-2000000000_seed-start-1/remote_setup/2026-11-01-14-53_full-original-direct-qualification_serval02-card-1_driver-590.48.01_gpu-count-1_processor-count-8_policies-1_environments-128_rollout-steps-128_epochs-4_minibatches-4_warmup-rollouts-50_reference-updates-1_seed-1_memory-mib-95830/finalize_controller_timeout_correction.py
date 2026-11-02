"""Retain the prior controller records and record only the reviewed timeout correction."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).absolute().parent
ARCHIVE = P / "controller_source_versions/before_timeout_correction_6782816_35"
REVIEW = P.parent / "2026-11-01-15-45_direct-qualification-controller_independent-review_private-cpu-only"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):
    return {"path": str(path), "sha256": sha(path)}
def raw_write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
def write(path, value):
    raw_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
def replace_record(path, raw):
    temporary = path.with_name(path.name + ".timeout-correction")
    raw_write(temporary, raw)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)

author_path = P / "controller_metadata_tests/author_6782816_36.json"
author = json.loads(author_path.read_text())
assert author["passed"] is True and author["tests_run"] == 19 and author["step_id"] == "36"
assert author["source_before"] == author["source_after"]
assert author["original_66_before"] == author["original_66_after"]
assert sha(REVIEW / "author_6782816_35.json") == "8028e29160025393a09d73852e6f429d4e30379e328d226c0256ee87b082e73a"
for source_name, copied_name in (
    ("author_6782816_35.json", "independent_timeout_finding_6782816_35.json"),
    ("guarded_independent_runner.py", "independent_timeout_finding_runner_6782816_35.py")):
    raw_write(P / "controller_metadata_tests" / copied_name, (REVIEW / source_name).read_bytes())
    (P / "controller_metadata_tests" / copied_name).chmod(0o444)

standard_archive = ARCHIVE / "controller_preparation"
standard_archive.mkdir()
for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
    raw_write(standard_archive / name, (P / "controller_preparation" / name).read_bytes())
    (standard_archive / name).chmod(0o444)
correction = {"schema_version": 1, "kind": "direct_qualification_controller_timeout_correction_author_review",
    "at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
    "original_packet_66_unchanged": True, "approved_44f_source_unchanged": True,
    "previous_27_file_freeze": receipt(ARCHIVE / "controller_source_freeze.json"),
    "independent_reproduction": receipt(P / "controller_metadata_tests/independent_timeout_finding_6782816_35.json"),
    "actual_author_verification": receipt(author_path), "actual_tests": 19, "actual_step": "6782816.36",
    "source": receipt(P / "direct_qualification_controller.py"), "tests": receipt(P / "test_direct_qualification_controller.py"),
    "before": "A signaled child could handle SIGTERM and return zero; the controller then published ordinary terminal success.",
    "after": "Any nonempty timeout/intervention list makes the controller fail, retains the actual integer return, writes a distinct wait kind and timeout observation, and publishes no ordinary terminal process record.",
    "actual_zero_return_preserved": True, "unchanged_terminal_reader_requires_an_ordinary_wait_kind": True,
    "natural_zero_positive_still_passes": True, "timeout_handler_zero_negative_passes": True,
    "no_original_producer_or_reader_change": True, "actual_target_contact": False, "actual_target_launch": False,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
write(P / "controller_timeout_correction_author_review.json", correction)

for name in ("manifest.yaml", "config_resolved.yaml"):
    path = P / "controller_preparation" / name
    value = json.loads(path.read_text())
    value["prior_metadata_verification"] = value["actual_metadata_verification"]
    value["actual_metadata_verification"] = receipt(author_path)
    value["timeout_correction"] = receipt(P / "controller_timeout_correction_author_review.json")
    value["controller"] = receipt(P / "direct_qualification_controller.py")
    value["tests"] = receipt(P / "test_direct_qualification_controller.py")
    if "metadata_test_work" in value:
        value["metadata_test_work"]["full_corrected_tests"] = 19
        value["metadata_test_work"]["step"] = "36"
        value["metadata_test_work"]["private_cpu_child_spawns"] = len(author["private_child_actions"])
    replace_record(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
background = P / "controller_preparation/experiment_background.md"
text = background.read_text()
text += "\nThe independent reviewer then reproduced one additional timeout defect on CPU step 6782816.35: a SIGTERM handler returned zero and the controller wrongly published success. The exact prior source, 27-file freeze and 19-case failure evidence are retained. The narrow correction treats any timeout or wait intervention as failure, preserves the actual zero return in a distinct wait record and never publishes an ordinary terminal process record for that attempt. All 19 corrected cases, including the natural zero positive and handled-timeout zero negative, passed on actual CPU step 6782816.36. The original 66 files and 44f producer remain unchanged.\n"
replace_record(background, text.encode())
write(P / "controller_timeout_corrected_standard_records.json", {
    "schema_version": 1, "kind": "timeout_corrected_controller_standard_records",
    "files": {name: receipt(P / "controller_preparation" / name)
        for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md")},
    "original_66_unchanged": True, "actual_target_contact": False})

previous = json.loads((ARCHIVE / "controller_source_freeze.json").read_text())
owned = {Path(name) for name in previous["files"]}
owned.update({P / "finalize_controller_timeout_correction.py",
              P / "controller_timeout_correction_author_review.json", P / "controller_timeout_corrected_standard_records.json"})
for directory in ("controller_metadata_tests", "controller_source_versions"):
    owned.update(path for path in (P / directory).rglob("*") if path.is_file())
frozen = {"schema_version": 1, "kind": "direct_qualification_controller_source_freeze",
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "original_packet_freeze": receipt(P / "packet_metadata_freeze.json"),
    "original_66_files_unchanged": True, "previous_controller_freeze": receipt(ARCHIVE / "controller_source_freeze.json"),
    "timeout_correction": receipt(P / "controller_timeout_correction_author_review.json"),
    "files": {str(path): sha(path) for path in sorted(owned)},
    "actual_target_contact": False, "actual_target_launch": False,
    "review_and_publication_required_before_launch": True,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
replace_record(P / "controller_source_freeze.json", (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode())
for name in ("direct_qualification_controller.py", "test_direct_qualification_controller.py",
             "controller_timeout_correction_author_review.json", "controller_source_freeze.json"):
    print(name, sha(P / name))
print("corrected controller freeze paths", len(owned))

