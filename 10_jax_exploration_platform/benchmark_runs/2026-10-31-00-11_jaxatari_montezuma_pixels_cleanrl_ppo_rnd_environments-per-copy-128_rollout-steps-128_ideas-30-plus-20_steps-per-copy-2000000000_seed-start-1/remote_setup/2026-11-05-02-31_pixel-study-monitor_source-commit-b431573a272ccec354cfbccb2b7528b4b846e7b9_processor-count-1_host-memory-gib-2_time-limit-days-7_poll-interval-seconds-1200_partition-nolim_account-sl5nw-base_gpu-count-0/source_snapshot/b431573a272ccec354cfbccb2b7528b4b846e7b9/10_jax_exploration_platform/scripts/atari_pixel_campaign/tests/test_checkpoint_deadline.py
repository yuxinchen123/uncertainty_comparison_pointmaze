"""Check proactive checkpoint timing using fake monotonic clocks and complete update counts."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_optimized as training
from tests import test_train_optimized as fixtures


class FakeClock:
    """Advance explicitly through complete training calls and durable checkpoint writes."""

    def __init__(self, now=0.0):
        """Start at a known monotonic timestamp without sleeping."""
        self.now = now

    def advance(self, seconds):
        """Include time spent writing instead of resetting the deadline before completion."""
        self.now += seconds
        return self.now

    def __call__(self):
        """Supply the same fake monotonic clock to the production timing helpers."""
        return self.now


class CheckpointDeadlineTests(unittest.TestCase):
    """Keep measured calls and serialization inside each available checkpoint window."""

    def policy(self, interval=1200, save_seconds=5):
        """Represent one initial complete durable save on the actual target."""
        return training.CheckpointDeadline(interval, 0, save_seconds)

    def calibrated(self, interval=120):
        """Supply explicit already-measured cold and warm timing observations."""
        policy = self.policy(interval)
        policy.cold_call_seconds = {1: 15, 3: 40}
        policy.maximum_warm_call_seconds = {1: 5, 3: 30}
        return policy

    def test_deadline_starts_after_completed_write(self):
        """A twenty-second initial write starts the next 120-second interval at completion."""
        policy = self.policy(120, 20)
        policy.maximum_warm_call_seconds[1] = 5
        self.assertEqual(policy.last_completed, 20)
        self.assertFalse(policy.plan(80, 1)["checkpoint_before_call"])
        self.assertTrue(policy.plan(90, 1)["checkpoint_before_call"])
        self.assertIsNone(policy.record()["completed_checkpoint_spacing_seconds"])

    def test_write_time_is_part_of_actual_spacing_and_violation(self):
        """Ninety seconds of work plus an eleven-second write exceeds a hundred-second interval."""
        clock, policy = FakeClock(5), self.policy(100)
        clock.advance(90)
        self.assertTrue(policy.checkpoint_due(clock.now))
        started = clock.now
        clock.advance(11)
        record = policy.completed_checkpoint(started, clock.now)
        self.assertEqual(record["completed_checkpoint_spacing_seconds"], 101)
        self.assertEqual(record["checkpoint_save_seconds"], 11)
        self.assertTrue(record["checkpoint_deadline_exceeded"])
        self.assertEqual(policy.last_completed, 106)

    def test_near_deadline_uses_one_original_update_when_group_does_not_fit(self):
        """A fitting single update uses the remaining time without padding the original schedule."""
        policy = self.calibrated()
        decision = policy.plan(60, 3)
        self.assertEqual(decision["updates"], 1)
        self.assertFalse(decision["checkpoint_before_call"])
        self.assertEqual(decision["call_allowance_seconds"], 10)

    def test_checkpoint_precedes_call_when_neither_count_fits_remaining_time(self):
        """The same selected group fits immediately after a new completed checkpoint."""
        policy = self.calibrated()
        self.assertTrue(policy.plan(100, 3)["checkpoint_before_call"])
        record = policy.completed_checkpoint(100, 105)
        self.assertFalse(record["checkpoint_deadline_exceeded"])
        decision = policy.plan(105, 3)
        self.assertEqual(decision["updates"], 3)
        self.assertFalse(decision["checkpoint_before_call"])

    def test_group_that_cannot_fit_full_interval_falls_back_to_single_updates(self):
        """A measured long group keeps the same learner updates through the original single-update path."""
        policy = self.calibrated()
        policy.maximum_warm_call_seconds[3] = 70
        decision = policy.plan(5, 3)
        self.assertEqual(decision["updates"], 1)
        self.assertFalse(decision["checkpoint_before_call"])

    def test_unfit_single_update_or_save_is_rejected_before_another_call(self):
        """No scheduler decision authorizes a call whose conservative single/save allowance cannot fit."""
        policy = self.calibrated()
        policy.maximum_warm_call_seconds[1] = 50
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            policy.plan(5, 3)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            self.policy(120, 61).plan(61, 3)

    def test_cold_six_hundred_second_compile_and_warm_thirty_second_group_remain_grouped(self):
        """Cold compilation is recorded separately and never permanently becomes the warm-call allowance."""
        clock, policy = FakeClock(5), self.policy()
        self.assertTrue(policy.plan(clock.now, 8)["unmeasured_call"])
        started = clock.now
        clock.advance(600)
        self.assertEqual(policy.observe_call(8, started, clock.now), "cold")
        started = clock.now
        clock.advance(5)
        self.assertFalse(policy.completed_checkpoint(started, clock.now)["checkpoint_deadline_exceeded"])
        decision = policy.plan(clock.now, 8)
        self.assertEqual(decision["updates"], 8)
        self.assertTrue(decision["unmeasured_call"])
        self.assertFalse(decision["checkpoint_before_call"])
        started = clock.now
        clock.advance(30)
        self.assertEqual(policy.observe_call(8, started, clock.now), "warm_calibration")
        started = clock.now
        clock.advance(5)
        policy.completed_checkpoint(started, clock.now)
        decision = policy.plan(clock.now, 8)
        self.assertEqual(decision["updates"], 8)
        self.assertFalse(decision["unmeasured_call"])
        self.assertEqual(decision["call_allowance_seconds"], 60)
        self.assertEqual(policy.record()["cold_call_seconds"], {"8": 600})
        self.assertEqual(policy.record()["maximum_observed_warm_call_seconds"], {"8": 30})

    def test_unmeasured_variant_gets_a_fresh_window_including_tail(self):
        """A new single-update tail is measured only after a fresh checkpoint rather than a used window."""
        policy = self.calibrated()
        policy.maximum_warm_call_seconds.pop(1)
        policy.cold_call_seconds.pop(1)
        policy.observe_call(3, 5, 35)
        decision = policy.plan(35, 1)
        self.assertEqual(decision["updates"], 1)
        self.assertTrue(decision["checkpoint_before_call"])
        policy.completed_checkpoint(35, 40)
        self.assertFalse(policy.plan(40, 1)["checkpoint_before_call"])

    def test_unexpected_long_warm_call_is_saved_and_reported_as_a_violation(self):
        """A measured bound is an allowance, and an actual longer stall is never reported as compliance."""
        clock, policy = FakeClock(5), self.calibrated()
        started = clock.now
        clock.advance(123)
        self.assertIsNone(policy.observe_call(3, started, clock.now))
        self.assertTrue(policy.checkpoint_due(clock.now))
        started = clock.now
        clock.advance(5)
        record = policy.completed_checkpoint(started, clock.now)
        self.assertEqual(record["completed_checkpoint_spacing_seconds"], 128)
        self.assertTrue(record["checkpoint_deadline_exceeded"])
        self.assertEqual(record["maximum_observed_warm_call_seconds"]["3"], 123)

    def test_cold_or_warm_overshoot_preserves_full_state_and_timing_before_failure_without_retry(self):
        """The production wrapper persists a real complete host checkpoint and evidence before rejecting lateness."""
        for phase in ("cold", "warm"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                path, timing_path = Path(temporary) / "latest.pkl", Path(temporary) / "checkpoint_timing.json"
                identity = fixtures.example_identity()
                report = {"pending_sums": None, "pending_count": 0, "last_report_iteration": 0}
                training.save_checkpoint(path, fixtures.example_state(3), report, identity, 3, 3.0)
                clock, policy, events = FakeClock(5), self.policy(), []
                if phase == "warm":
                    policy.cold_call_seconds[1] = 600
                    policy.maximum_warm_call_seconds[1] = 30
                self.assertFalse(policy.plan(clock.now, 1)["checkpoint_before_call"])
                started = clock.now
                clock.advance(1201)
                policy.observe_call(1, started, clock.now)
                def save():
                    """Write every state field once, including the actual next iteration."""
                    training.save_checkpoint(path, fixtures.example_state(4), report, identity, 4, 1204.0)
                    clock.advance(5)
                    events.append("state")
                def record(timing):
                    """Publish explicit timing evidence after the complete pair already exists."""
                    self.assertEqual(training.load_checkpoint(path, identity)["iteration"], 4)
                    training.atomic_json(timing_path, timing)
                    events.append("timing")
                with self.assertRaisesRegex(RuntimeError, "full state was saved and requires diagnosis"):
                    training.checkpoint_with_timing(save, record, 1200, policy, clock)
                self.assertEqual(events, ["state", "timing"])
                self.assertEqual(training.load_checkpoint(path, identity)["state"], fixtures.example_state(4))
                timing = json.loads(timing_path.read_text())
                self.assertEqual(timing["completed_checkpoint_spacing_seconds"], 1206)
                self.assertTrue(timing["checkpoint_deadline_exceeded"])
                self.assertEqual(timing["checkpoint_save_seconds"], 5)

    def test_production_wrapper_counts_initial_and_later_writes_at_completion(self):
        """The first save establishes the deadline and a later write contributes to its measured spacing."""
        clock, records = FakeClock(), []
        policy = training.checkpoint_with_timing(lambda: clock.advance(7), records.append, 1200, clock=clock)
        self.assertIsNone(records[-1]["completed_checkpoint_spacing_seconds"])
        self.assertEqual(policy.last_completed, 7)
        clock.advance(100)
        same = training.checkpoint_with_timing(lambda: clock.advance(17), records.append, 1200, policy, clock)
        self.assertIs(same, policy)
        self.assertEqual(records[-1]["completed_checkpoint_spacing_seconds"], 117)
        self.assertEqual(records[-1]["checkpoint_save_seconds"], 17)
        self.assertFalse(records[-1]["checkpoint_deadline_exceeded"])

    def test_stop_requested_during_pre_call_save_prevents_another_training_call(self):
        """A signal received while writing is honored once that complete checkpoint returns."""
        clock, policy, stop = FakeClock(100), self.calibrated(), {"reason": None}
        calls = []
        def checkpoint(reason):
            """Finish the requested save and represent SIGUSR1 arriving during its write."""
            calls.append(reason)
            started = clock.now
            clock.advance(5)
            policy.completed_checkpoint(started, clock.now)
            stop["reason"] = "SIGUSR1"
        count = policy.prepare_call(3, checkpoint, lambda: stop["reason"] is not None, clock)
        self.assertIsNone(count)
        self.assertEqual(calls, ["before_training_call"])
        self.assertEqual(stop["reason"], "SIGUSR1")

    def test_process_time_limit_rechecked_after_pre_call_save(self):
        """A save that reaches the process time limit does not authorize one more group."""
        clock, policy, calls = FakeClock(100), self.calibrated(), []
        def checkpoint(reason):
            """Consume the remaining wall time while retaining a complete checkpoint."""
            calls.append(reason)
            started = clock.now
            clock.advance(5)
            policy.completed_checkpoint(started, clock.now)
        self.assertIsNone(policy.prepare_call(3, checkpoint, lambda: clock.now >= 105, clock))
        self.assertEqual(calls, ["before_training_call"])

    def test_fallback_and_tail_preserve_every_original_update_and_report(self):
        """A mixed call sequence retains the same full fake state and unfinished report as twenty original updates."""
        def advance(state, count):
            """Represent complete ordered updates with one report per four updates."""
            for _ in range(count):
                state["iteration"] += 1
                state["value"] = 31 * state["value"] + state["iteration"]
                state["pending"].append(state["value"])
                if state["iteration"] % 4 == 0:
                    state["reports"].append(tuple(state["pending"]))
                    state["pending"] = []
        initial = {"iteration": 0, "value": 7, "pending": [], "reports": []}
        reference = copy.deepcopy(initial)
        advance(reference, 20)
        state, clock, policy, counts = copy.deepcopy(initial), FakeClock(60), self.calibrated(), []
        while state["iteration"] < 20:
            preferred = training.next_update_count(state["iteration"], 20, 3)
            decision = policy.plan(clock.now, preferred)
            if decision["checkpoint_before_call"]:
                started = clock.now
                clock.advance(5)
                self.assertFalse(policy.completed_checkpoint(started, clock.now)["checkpoint_deadline_exceeded"])
                decision = policy.plan(clock.now, preferred)
                self.assertFalse(decision["checkpoint_before_call"])
            count, started = decision["updates"], clock.now
            advance(state, count)
            clock.advance(30 if count == 3 else 5)
            policy.observe_call(count, started, clock.now)
            counts.append(count)
            if policy.checkpoint_due(clock.now):
                started = clock.now
                clock.advance(5)
                self.assertFalse(policy.completed_checkpoint(started, clock.now)["checkpoint_deadline_exceeded"])
        self.assertEqual(state, reference)
        self.assertEqual(sum(counts), 20)
        self.assertIn(3, counts)
        self.assertIn(1, counts)

    def test_invalid_clock_values_and_counts_are_rejected(self):
        """Reversed, nonfinite and Boolean observations cannot become timing evidence."""
        for values in ((0, float("nan")), (10, 9), (False, 2), (-1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                training.CheckpointDeadline.duration(*values)
        policy = self.policy()
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                policy.plan(5, count)
        with self.assertRaises(ValueError):
            policy.plan(4, 1)


if __name__ == "__main__":
    unittest.main()
