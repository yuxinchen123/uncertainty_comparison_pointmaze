"""The submission plan of the maze campaign (no job is submitted), its job scripts, and the cancel helper's refusals."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from maze_test_support import MAZE
from grid_campaign import own

submit = own("submit")
cancel_ledger = own("cancel_ledger")

CLASSES = [
    {"gres_type": "h100", "partition": "gpu", "nodes": ["serval06"], "gpus_per_node": 2, "gpu_mem_mb": 95830,
     "capability_rank": 1},
    {"gres_type": "rtx5080", "partition": "gpu", "nodes": ["nekomata01", "nekomata02"], "gpus_per_node": 2,
     "gpu_mem_mb": 16303, "capability_rank": 3},
    {"gres_type": "a4500", "partition": "gpu", "nodes": ["jaguar03"], "gpus_per_node": 8, "gpu_mem_mb": 20470,
     "capability_rank": 6},
    {"gres_type": "a16", "partition": "gpu", "nodes": ["hydro"], "gpus_per_node": 8, "gpu_mem_mb": 15356,
     "capability_rank": 8},
    {"gres_type": "gtx1080", "partition": "gnolim", "nodes": ["ai05"], "gpus_per_node": 4, "gpu_mem_mb": 8192,
     "capability_rank": 15},
]
NODES = {
    "serval06": {"cards": 0, "cpus": 64, "memory_gib": 500, "reservation": None},
    "nekomata01": {"cards": 2, "cpus": 32, "memory_gib": 100, "reservation": None},
    "nekomata02": {"cards": 1, "cpus": 32, "memory_gib": 100, "reservation": "(null)"},
    "jaguar03": {"cards": 5, "cpus": 200, "memory_gib": 900, "reservation": "sl5nw_168"},
    "hydro": {"cards": 8, "cpus": 64, "memory_gib": 500, "reservation": "as8hu_169"},
    "ai05": {"cards": 4, "cpus": 16, "memory_gib": 64, "reservation": None},
}
BUDGETS = {(identity, partition): {"cards": cards, "cpus": cpus, "memory_gib": 1000}
           for identity in submit.IDENTITIES for partition, cards, cpus in (("gpu", 40, 384), ("gnolim", 20, 80))}
RESERVED = {"jaguar03": {"cards": 5, "cpus": 184, "memory_gib": 900, "name": "sl5nw_168"}}
# cards of each type on nodes outside every reservation: the RTX A4500s are all inside the owner's
# reservation and the A16s inside another user's
ROOM = {"h100": 2, "rtx5080": 4, "a4500": 0, "a16": 0, "gtx1080": 4}


def plan(units, without_unit=0, **changes):
    """Plan with the fixture above; keyword arguments replace single inputs."""
    inputs = {"classes": CLASSES, "nodes": NODES, "budgets": BUDGETS, "reservation_budgets": RESERVED,
              "held_nodes": set(), "type_room": ROOM, "waiting": {}}
    extra = {key: changes.pop(key) for key in ("allowed_types", "max_new_cards", "max_waiting") if key in changes}
    inputs.update(changes)
    return submit.plan_jobs(units, without_unit, inputs["classes"], inputs["nodes"], inputs["budgets"],
                            inputs["reservation_budgets"], inputs["held_nodes"], inputs["type_room"],
                            inputs["waiting"], **extra)


class Plan(unittest.TestCase):
    """Which worker jobs one submission sends."""

    def test_open_nodes_first_then_the_reservation_then_waiting_jobs(self):
        """Pinned jobs on nodes with free cards, the reserved node last, then one-card jobs that wait, by card type."""
        jobs = plan([7000] * 20)
        pinned = [(job["node"], job["cards"], job["reservation"]) for job in jobs if job["node"]]
        self.assertEqual(pinned, [("nekomata01", 2, None), ("nekomata02", 1, None), ("ai05", 4, None),
                                  ("jaguar03", 5, "sl5nw_168")])
        self.assertEqual([job["identity"] for job in jobs if job["node"] == "jaguar03"], ["uva-sl5nw"])
        waiting = [(job["gpu_type"], job["identity"]) for job in jobs if not job["node"]]
        # open cards left per type: 2 H100; 4 - 3 RTX 5080; no GTX 1080; the reserved types get no waiting job
        self.assertEqual(waiting, [("h100", "uva-sl5nw"), ("h100", "uva-yuxinchen"), ("rtx5080", "uva-sl5nw")])
        self.assertTrue(all(job["cards"] == 1 and job["reservation"] is None for job in jobs if not job["node"]))
        self.assertLessEqual(sum(job["cards"] for job in jobs), 20)
        # 8 processors per card in gpu, 4 in gnolim, 16 GiB per card
        self.assertEqual([(job["cpus"], job["memory_gib"]) for job in jobs[:3]], [(16, 32), (8, 16), (16, 64)])

    def test_a_smaller_request_per_card_fits_a_node_with_little_free_memory(self):
        """With 2 processors and 4 GiB per card, a node with 6 GiB of free memory takes one card; by default none."""
        nodes = deepcopy(NODES)
        nodes["nekomata01"]["memory_gib"] = 6
        self.assertNotIn("nekomata01", [job["node"] for job in plan([7000] * 2, nodes=nodes)])
        defaults = (dict(submit.CPUS_PER_CARD), submit.MEMORY_GIB_PER_CARD)
        try:
            submit.set_request_per_card(cpus=2, memory_gib=4)
            first = plan([7000] * 2, nodes=nodes)[0]
            self.assertEqual((first["node"], first["cards"], first["cpus"], first["memory_gib"]), ("nekomata01", 1, 2, 4))
        finally:
            submit.CPUS_PER_CARD.update(defaults[0])
            submit.set_request_per_card(memory_gib=defaults[1])

    def test_a_worker_time_limit_caps_worker_jobs_only(self):
        """A 2-hour limit shortens a worker's 4 days to 7,200 s, never lengthens, and leaves the monitor alone."""
        self.assertEqual(submit.wall_seconds("worker", 345600), 345600)
        try:
            submit.set_worker_hours(2)
            self.assertEqual(submit.wall_seconds("worker", 345600), 7200)
            self.assertEqual(submit.wall_seconds("worker", 3600), 3600)
            self.assertEqual(submit.wall_seconds("monitor", 345600), 345600)
            submit.set_worker_hours(0)
            with self.assertRaises(ValueError):
                submit.wall_seconds("worker", 345600)
        finally:
            submit.set_worker_hours(None)

    def test_new_workers_can_be_limited_to_one_account(self):
        """With one account named, every planned job is that account's; an unknown name is refused."""
        both = {job["identity"] for job in plan([7000] * 6)}
        self.assertEqual(both, {"uva-sl5nw", "uva-yuxinchen"})
        try:
            submit.set_identities(["uva-sl5nw"])
            self.assertEqual({job["identity"] for job in plan([7000] * 6)}, {"uva-sl5nw"})
            with self.assertRaises(ValueError):
                submit.set_identities(["uva-someone-else"])
        finally:
            submit.IDENTITIES = submit.ALL_IDENTITIES

    def test_jobs_that_wait_never_exceed_the_units_no_job_covers(self):
        """Units of 30 GB find no card free now: 3 units and 1 waiting worker of this run give 2 new cards, 3 give none."""
        self.assertEqual(sum(job["cards"] for job in plan([30000] * 3, without_unit=1)), 2)
        self.assertEqual(plan([30000] * 3, without_unit=3), [])
        self.assertEqual(plan([]), [])
        self.assertEqual(sum(job["cards"] for job in plan([7000] * 20, max_new_cards=1)), 1)

    def test_cards_free_now_take_units_that_waiting_jobs_already_cover(self):
        """Three units and three waiting workers of this run: the 3 cards free now on open nodes still get jobs."""
        jobs = plan([7000] * 3, without_unit=3)
        self.assertEqual([(job["node"], job["cards"]) for job in jobs], [("nekomata01", 2), ("nekomata02", 1)])

    def test_waiting_high_end_jobs_leave_free_lower_cards_usable(self):
        """2 units, 2 H100 jobs of this run waiting, 3 RTX 2080 Ti free now: 2 pinned one-card jobs, no new waiting job."""
        classes = [CLASSES[0], {"gres_type": "rtx2080ti", "partition": "gpu", "nodes": ["ai01", "ai02", "ai03"],
                                "gpus_per_node": 4, "gpu_mem_mb": 11264, "capability_rank": 12}]
        nodes = {"serval06": {"cards": 0, "cpus": 64, "memory_gib": 500, "reservation": None},
                 **{name: {"cards": 1, "cpus": 16, "memory_gib": 64, "reservation": None} for name in ("ai01", "ai02", "ai03")}}
        # the 2 waiting H100 jobs already ask for both open H100 cards
        inputs = dict(classes=classes, nodes=nodes, reservation_budgets={}, type_room={"h100": 0, "rtx2080ti": 12},
                      waiting={"uva-sl5nw": 2})
        jobs = plan([7000] * 2, without_unit=2, **inputs)
        self.assertEqual([(job["node"], job["cards"], job["gpu_type"]) for job in jobs],
                         [("ai01", 1, "rtx2080ti"), ("ai02", 1, "rtx2080ti")])
        # golden path: 5 units and 1 waiting job: 3 cards free now, then 1 new job that waits for the fifth unit
        jobs = plan([7000] * 5, without_unit=1, **dict(inputs, type_room={"h100": 2, "rtx2080ti": 12}))
        self.assertEqual([(job["node"], job["gpu_type"]) for job in jobs],
                         [("ai01", "rtx2080ti"), ("ai02", "rtx2080ti"), ("ai03", "rtx2080ti"), (None, "h100")])
        # a limit on new cards counts the pinned ones first
        self.assertEqual([job["node"] for job in plan([7000] * 5, without_unit=1, max_new_cards=2, **inputs)],
                         ["ai01", "ai02"])

    def test_free_cards_of_a_type_that_waiting_jobs_already_ask_for_still_get_jobs(self):
        """2 units; this run's 2 waiting A100 jobs wait on one account's cap and use up the type's room; 2 A100 free now."""
        classes = [{"gres_type": "a100_40gb", "partition": "gpu", "nodes": ["cheetah01"], "gpus_per_node": 4,
                    "gpu_mem_mb": 40960, "capability_rank": 2}]
        nodes = {"cheetah01": {"cards": 2, "cpus": 24, "memory_gib": 230, "reservation": None}}
        budgets = {**BUDGETS, ("uva-sl5nw", "gpu"): {"cards": 0, "cpus": 0, "memory_gib": 0}}
        jobs = plan([7000] * 2, without_unit=2, classes=classes, nodes=nodes, budgets=budgets, reservation_budgets={},
                    type_room={"a100_40gb": 0}, waiting={"uva-sl5nw": 2})
        self.assertEqual([(job["identity"], job["node"], job["cards"]) for job in jobs],
                         [("uva-yuxinchen", "cheetah01", 2)])
        # no new job waits for the type, since its room is used up
        jobs = plan([7000] * 5, without_unit=2, classes=classes, nodes=nodes, budgets=budgets, reservation_budgets={},
                    type_room={"a100_40gb": 0}, waiting={"uva-sl5nw": 2})
        self.assertEqual([(job["node"], job["cards"]) for job in jobs], [("cheetah01", 2)])

    def test_only_cards_that_can_hold_a_pending_unit_are_asked_for(self):
        """Units that need 30 GB go to the H100 class alone, which has no free card: two jobs wait."""
        jobs = plan([30000] * 5)
        self.assertEqual([(job["gpu_type"], job["node"]) for job in jobs], [("h100", None), ("h100", None)])
        self.assertEqual(plan([200000] * 5), [])

    def test_allowed_types_and_other_peoples_reservations(self):
        """A type that is not allowed gets no job; a node reserved for someone else gets none either."""
        jobs = plan([7000] * 20, allowed_types=["rtx5080", "a16"])
        self.assertEqual({job["gpu_type"] for job in jobs}, {"rtx5080"})
        # even with room for the type, the node inside another user's reservation is passed over
        jobs = plan([7000] * 20, allowed_types=["rtx5080", "a16"], type_room=dict(ROOM, a16=8), max_waiting=0)
        self.assertEqual([job["node"] for job in jobs], ["nekomata01", "nekomata02"])

    def test_budgets_held_nodes_and_the_limit_on_waiting_jobs(self):
        """The per-user room, a node where a job of ours still waits, and 40 waiting jobs per account bound the plan."""
        tight = deepcopy(BUDGETS)
        tight[("uva-sl5nw", "gpu")]["cards"] = 0
        tight[("uva-yuxinchen", "gpu")]["cards"] = 1
        jobs = plan([7000] * 20, budgets=tight, reservation_budgets={}, allowed_types=["rtx5080"], max_waiting=0)
        self.assertEqual([(job["identity"], job["node"], job["cards"]) for job in jobs],
                         [("uva-yuxinchen", "nekomata01", 1)])
        jobs = plan([7000] * 20, held_nodes={"nekomata01"}, allowed_types=["rtx5080"], max_waiting=0)
        self.assertEqual([job["node"] for job in jobs], ["nekomata02"])
        jobs = plan([30000] * 5, waiting={"uva-sl5nw": 40, "uva-yuxinchen": 39})
        self.assertEqual([(job["identity"], job["node"]) for job in jobs], [("uva-yuxinchen", None)])

    def test_the_plan_changes_none_of_its_inputs(self):
        """Planning twice gives the same jobs."""
        before = deepcopy((NODES, BUDGETS, RESERVED, ROOM))
        self.assertEqual(plan([7000] * 20), plan([7000] * 20))
        self.assertEqual((NODES, BUDGETS, RESERVED, ROOM), before)


class CardTypesFromTheScheduler(unittest.TestCase):
    """A node is planned under the card type the scheduler reports for it now."""

    LISTING = ("NodeName=lynx02 Arch=x86_64 CPUTot=32 Gres=gpu:nvidia_rtx_4000_ada_generation:4(S:0-1) State=MIXED\n"
               "NodeName=ai01 Arch=x86_64 CPUTot=32 Gres=gpu:nvidia_geforce_rtx_2080_ti:4(S:0-1) State=IDLE\n"
               "NodeName=cortado01 Arch=x86_64 CPUTot=24 Gres=(null) State=IDLE\n")

    def test_live_card_types(self):
        """One type per node that holds typed cards; a node without cards is left out."""
        self.assertEqual(submit.live_card_types(self.LISTING),
                         {"lynx02": "nvidia_rtx_4000_ada_generation", "ai01": "nvidia_geforce_rtx_2080_ti"})

    def test_a_node_with_replaced_cards_moves_to_the_class_of_its_live_type(self):
        """Moved within its partition, left out when no class has the type, kept when the scheduler lists no type."""
        classes = [{"gres_type": "a", "partition": "gpu", "nodes": ["n1", "n2", "n4", "n5"]},
                   {"gres_type": "b", "partition": "gnolim", "nodes": ["n6"]},
                   {"gres_type": "b", "partition": "gpu", "nodes": ["n3"]}]
        before = deepcopy(classes)
        result = submit.classes_as_the_scheduler_has_them(classes, {"n1": "a", "n2": "b", "n3": "c", "n5": "c"})
        self.assertEqual(result, [{"gres_type": "a", "partition": "gpu", "nodes": ["n1", "n4"]},
                                  {"gres_type": "b", "partition": "gnolim", "nodes": ["n6"]},
                                  {"gres_type": "b", "partition": "gpu", "nodes": ["n2"]}])
        self.assertEqual(classes, before)

    def test_only_cards_free_now_sends_no_job_that_would_wait(self):
        """With the limit of waiting jobs at zero every planned job is pinned to a node."""
        jobs = plan([7000] * 20, max_waiting=0)
        self.assertTrue(jobs)
        self.assertTrue(all(job["node"] for job in jobs))
        self.assertTrue(any(job["node"] is None for job in plan([7000] * 20)))

    def test_only_jobs_that_wait_pins_no_node(self):
        """A dry run for 4 pending units: by default pinned jobs come first; with only_waiting every job waits by type."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            for state in ("pending", "running", "done", "failed"):
                (run_dir / "queue" / state).mkdir(parents=True)
            for index in range(4):
                (run_dir / "queue/pending" / f"unit-{index}.json").write_text(
                    f'{{"name": "unit-{index}", "order": {index}, "minimum_device_memory_mb": 7000}}\n')
            state = {"classes": CLASSES, "nodes": NODES, "budgets": BUDGETS, "reservation_budgets": RESERVED,
                     "held_nodes": set(), "type_room": ROOM, "waiting": {}, "waiting_cards": 0}
            originals = {name: getattr(submit, name) for name in ("frozen_code", "scheduler_state")}
            printed = {}
            try:
                submit.frozen_code = lambda run: (Path("/p/code"), "abc")
                submit.scheduler_state = lambda run: deepcopy(state)
                for only_waiting in (False, True):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        submit.top_up(run_dir, "development", dry_run=True, only_waiting=only_waiting)
                    printed[only_waiting] = [line for line in output.getvalue().splitlines()
                                             if line.startswith("  uva-")]
                with self.assertRaises(ValueError):
                    submit.top_up(run_dir, "development", dry_run=True, only_free_now=True, only_waiting=True)
            finally:
                for name, function in originals.items():
                    setattr(submit, name, function)
        # before: 2 RTX 5080 cards on nekomata01, 1 on nekomata02 (the account with more room left), 1 GTX 1080 on
        # ai05, all pinned
        self.assertEqual(printed[False], ["  uva-sl5nw gpu nekomata01 2 x rtx5080",
                                          "  uva-yuxinchen gpu nekomata02 1 x rtx5080",
                                          "  uva-sl5nw gnolim ai05 1 x gtx1080"])
        # after: one-card jobs by type that wait, the most capable type first, shared between the accounts
        self.assertEqual(printed[True], ["  uva-sl5nw gpu any node 1 x h100", "  uva-yuxinchen gpu any node 1 x h100",
                                         "  uva-sl5nw gpu any node 1 x rtx5080", "  uva-yuxinchen gpu any node 1 x rtx5080"])

    def test_waiting_jobs_of_the_run_leave_the_room_for_cards_free_now(self):
        """An account with 3 cards of room and 4 waiting jobs of this run keeps its 3 cards for cards free now."""
        from unittest import mock
        capacity = {identity: {"reachable": True, "pools": [
            {"partition": "gpu", "gpu_free": 3, "cpu_free": 200, "memory_gib_free": 900}]} for identity in submit.IDENTITIES}
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / "slurm/submissions").mkdir(parents=True)
            live = {}
            # before: 4 waiting one-card worker jobs of this run under every account
            for identity in submit.IDENTITIES:
                for index in range(4):
                    job_id = f"{identity[-5:]}{index}"
                    live[job_id] = [job_id, identity, "gpu", "PENDING"]
                    (run_dir / "slurm/submissions" / f"{job_id}.json").write_text(json.dumps(
                        {"job_id": job_id, "kind": "worker", "account": identity.removeprefix("uva-"),
                         "identity": identity, "partition": "gpu", "cards": 1, "cpus": 4, "memory_gib": 10,
                         "gpu_type": "a40", "node": None, "reservation": None}))
            # the scheduler's answers: the capacity reader, an empty node listing, one A40 class on one node
            calls = lambda command, text: json.dumps(capacity) if "capacity.py" in command[1] else ""
            a40 = {"gres_type": "a40", "partition": "gpu", "nodes": ["n1"], "gpus_per_node": 4, "gpu_mem_mb": 46068,
                   "capability_rank": 5}
            read = lambda path: ({"classes": [a40]} if path == submit.grid_submit.RESOURCE_FILE
                                 else json.loads(Path(path).read_text()))
            node = {"n1": {"cards": 1, "cpus": 40, "memory_gib": 900, "reservation": None}}
            with mock.patch.object(submit.common, "own_live_jobs", return_value=live), \
                    mock.patch.object(submit.subprocess, "check_output", side_effect=calls), \
                    mock.patch.object(submit.common, "read_json", side_effect=read), \
                    mock.patch.object(submit.grid_submit, "node_resources", return_value=node), \
                    mock.patch.object(submit.grid_submit, "owner_reservation", return_value=None):
                state = submit.scheduler_state(run_dir)
        # after: every account keeps its 3 cards of room; the 8 waiting cards still count as spare workers
        self.assertEqual({key: budget["cards"] for key, budget in state["budgets"].items()},
                         {(identity, "gpu"): 3 for identity in submit.IDENTITIES})
        self.assertEqual(state["waiting_cards"], 4 * len(submit.IDENTITIES))

    def test_a_refused_job_does_not_stop_the_others(self):
        """top_up sends the remaining jobs after one refusal and returns only the ids that were sent."""
        jobs = [{"identity": "uva-sl5nw", "partition": "gpu", "node": name, "gpu_type": "a", "cards": 1,
                 "reservation": None} for name in ("refused", "accepted")]
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / "slurm").mkdir()
            for state in ("pending", "running", "done", "failed"):
                (run_dir / "queue" / state).mkdir(parents=True)

            def send(run, phase, job, text, dry_run=False):
                if job["node"] == "refused":
                    raise subprocess.CalledProcessError(1, ["sbatch"])
                with submit.ledger(run, phase, "sl5nw").open("a") as handle:
                    handle.write("4242\n")
                return "4242"

            originals = {name: getattr(submit, name) for name in
                         ("frozen_code", "scheduler_state", "plan_jobs", "submit_job", "worker_job_text")}
            state = {key: None for key in ("classes", "nodes", "budgets", "reservation_budgets", "held_nodes",
                                           "type_room", "waiting")}
            try:
                submit.frozen_code = lambda run: (Path("/p/code"), "abc")
                submit.scheduler_state = lambda run: dict(state, waiting_cards=0)
                submit.plan_jobs = lambda *arguments: jobs
                submit.submit_job = send
                submit.worker_job_text = lambda *arguments: "#!/bin/bash\n"
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    sent = submit.top_up(run_dir, "development")
            finally:
                for name, function in originals.items():
                    setattr(submit, name, function)
        self.assertEqual(sent, ["4242"])
        self.assertIn("NOT SUBMITTED", printed.getvalue())


class Scripts(unittest.TestCase):
    """The text of the job scripts and the names of the ledgers."""

    def test_worker_job_script(self):
        """One worker.py per card with the run folder and the observed folder; bash accepts the script."""
        text = submit.worker_job_text(Path("/p/run"), Path("/p/run/code/source_abc/10_jax_exploration_platform"),
                                      Path("/p/env/bin/python"), 2)
        self.assertIn("/p/env/bin/python /p/run/code/source_abc/10_jax_exploration_platform/scripts/maze_campaign/"
                      "worker.py --run-dir /p/run --observed-dir", text)
        self.assertIn("source /p/run/slurm/worker_env.sh", text)
        self.assertIn('-eq 2 ]', text)
        for script in (text, submit.monitor_job_text(Path("/p/run"), Path("/p/code"), Path("/p/env/bin/python"),
                                                     ["--top-up-phase", "development"])):
            self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(["bash", "-n"], input=submit.WORKER_ENVIRONMENT, text=True).returncode, 0)

    def test_ledger_and_python(self):
        """The ledger is named by phase and account; the Python comes from the environment registry."""
        self.assertEqual(submit.ledger("/p/run", "development-first-pass", "yuxinchen"),
                         Path("/p/run/slurm/submitted_jobids_development-first-pass_yuxinchen.txt"))
        self.assertEqual(submit.canonical_python().parts[-3:], ("platform_jax", "bin", "python"))

    def test_jobs_need_a_frozen_code_copy(self):
        """Without RUN/source.json nothing is planned or submitted."""
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit):
            submit.frozen_code(folder)


class Cancelling(unittest.TestCase):
    """This folder cancels nothing except through cancel_ledger.py, and that only for a stopped run's own ledger."""

    def test_no_other_file_holds_a_cancel_command(self):
        """No file names the scheduler's cancel command; only cancel_ledger.py calls the shared cancel wrappers."""
        for path in MAZE.glob("*.py"):
            text = path.read_text()
            self.assertNotIn("scancel", text, path.name)
            if path.name != "cancel_ledger.py":
                self.assertNotIn("cancel_as.sh", text, path.name)
                self.assertNotIn("cancel_pending(", text, path.name)

    def test_refusals_and_the_dry_run(self):
        """No STOP file, a file that is no ledger, and a server's ledger without --server are refused."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder).resolve()
            (run_dir / "slurm").mkdir()
            ledger = run_dir / "slurm/submitted_jobids_development_yuxinchen.txt"
            ledger.write_text("123\n")
            (run_dir / "slurm/submitted_jobids_mit.txt").write_text("9\n")
            (run_dir / "slurm/notes.txt").write_text("x\n")
            with self.assertRaises(SystemExit) as refusal:
                cancel_ledger.main(["--ledger", str(ledger), "--dry-run"])
            self.assertIn("STOP", str(refusal.exception))
            (run_dir / "STOP").write_text("stop\n")
            for path in (run_dir / "slurm/notes.txt", run_dir / "slurm/submitted_jobids_mit.txt",
                         run_dir / "slurm/submitted_jobids_absent_sl5nw.txt"):
                with self.assertRaises(SystemExit):
                    cancel_ledger.main(["--ledger", str(path), "--dry-run"])
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cancel_ledger.main(["--ledger", str(ledger), "--dry-run"])
            self.assertTrue(printed.getvalue().strip().endswith(f"cancel_as.sh uva-yuxinchen --idfile {ledger} --all"))
            # single ids of the ledger need no STOP file: a pinned job that still waits is re-placed this way
            (run_dir / "STOP").unlink()
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cancel_ledger.main(["--ledger", str(ledger), "--job-id", "123", "--dry-run"])
            self.assertTrue(printed.getvalue().strip().endswith(f"cancel_as.sh uva-yuxinchen --idfile {ledger} 123"))


if __name__ == "__main__":
    unittest.main()
