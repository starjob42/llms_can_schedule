"""Meaningful evaluator tests; run with python3 -m unittest discover -s experiments/phi3_revision -p 'test_core.py'."""
import math
import unittest

try:
    from .core import (
        EvaluationRecord, Instance, Operation, PROMPTS, aggregate_results,
        canonical_hash, parse_instance, parse_matrix, render_user_prompt,
        select_validation_prompt, validate_schedule, wilson_interval,
    )
except ImportError:
    from core import (
        EvaluationRecord, Instance, Operation, PROMPTS, aggregate_results,
        canonical_hash, parse_instance, parse_matrix, render_user_prompt,
        select_validation_prompt, validate_schedule, wilson_interval,
    )


MATRIX = "2 2\n0 3 1 2\n1 2 0 4\n7.0\n0 3\n2 1\n"
SCHEDULE = """Solution:
Job 0 Operation 0 on Machine 0 : 0 + 3 -> 3
Job 1 Operation 0 on Machine 1 : 0 + 2 -> 2
Job 0 Operation 1 on Machine 1 : 3 + 2 -> 5
Job 1 Operation 1 on Machine 0 : 3 + 4 -> 7
Makespan: 7.0, as it is the maximum end completion time.
"""
SHORT_SCHEDULE = "Solution:\nJ0-M0: 0+3 ->3,J1-M1: 0+2 ->2,J0-M1: 3+2 ->5,J1-M0: 3+4 ->7,\nMaximum end completion time or Makespan: 7\n"
MACHINE_TEXT = """Optimize schedule for 2 Jobs across 2 Machines to minimize makespan.
Machine 1 is used for the following Operations:
Job 0 Operation 1 duration 2 mins.
Job 1 Operation 0 duration 2 mins.
Machine 0 is used for the following Operations:
Job 1 Operation 1 duration 4 mins.
Job 0 Operation 0 duration 3 mins.
"""
JOB_TEXT = """Optimize schedule for 2 Jobs across 2 Machines to minimize makespan.
Job 1 consists of the following Operations:
Operation 1 on Machine 0 duration 4 mins.
Operation 0 on Machine 1 duration 2 mins.
Job 0 consists of the following Operations:
Operation 1 on Machine 1 duration 2 mins.
Operation 0 on Machine 0 duration 3 mins.
"""
COMPACT_JOB_TEXT = """Optimize schedule for 2 Jobs (denoted as J) across 2 Machines (denoted as M) to minimize makespan.
J0:
M0:3 M1:2
J1:
M1:2 M0:4
"""


class InstanceTests(unittest.TestCase):
    def test_matrix_reference_does_not_change_identity(self):
        instance = parse_matrix(MATRIX)
        self.assertEqual(instance.reference_makespan, 7)
        self.assertEqual(len(instance.operations), 4)
        self.assertEqual(canonical_hash(instance), canonical_hash(parse_matrix(MATRIX.replace("7.0", "99.0"))))

    def test_representation_and_row_order_independent_identity(self):
        matrix = parse_matrix(MATRIX)
        for text in (MACHINE_TEXT, JOB_TEXT, COMPACT_JOB_TEXT):
            self.assertEqual(canonical_hash(matrix), canonical_hash(parse_instance(text)))
        flat = "\n".join(f"Machine {o.machine} processes Job {o.job} Operation {o.operation} for {o.duration} mins."
                         for o in reversed(matrix.operations))
        self.assertEqual(canonical_hash(matrix), canonical_hash(parse_instance(flat)))

    def test_route_and_duration_change_identity(self):
        self.assertNotEqual(canonical_hash(parse_matrix(MATRIX)), canonical_hash(parse_matrix(MATRIX.replace("0 3 1 2", "1 3 0 2"))))
        self.assertNotEqual(canonical_hash(parse_matrix(MATRIX)), canonical_hash(parse_matrix(MATRIX.replace("0 3 1 2", "0 4 1 2"))))

    def test_compact_machine_input_cannot_invent_precedence(self):
        with self.assertRaisesRegex(ValueError, "omits precedence"):
            parse_instance("M0:\nJ0:3 J1:4\nM1:\nJ1:2 J0:2")

    def test_matrix_and_instance_reject_corruption(self):
        for matrix in ("", "2 2\n0 3 1 2", "1 2\n0 3", "1 1\n2 3", "1 1\n0 -3", "1 1\n0 1e999"):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                parse_matrix(matrix)
        for text in (MACHINE_TEXT + "Job 0 Operation 0 duration 3 mins.",
                     MACHINE_TEXT.replace("Operation 0 duration 3", "Operation 2 duration 3"),
                     MACHINE_TEXT.replace("Operation 0 duration 3", "Operation 0 duration BAD")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_instance(text)

    def test_duplicate_repeated_machine_routes_are_preserved(self):
        # Non-classical routes remain valid canonical instances, but compact
        # outputs without operation indices cannot disambiguate revisits.
        instance = parse_matrix("1 2\n0 3 0 4")
        result = validate_schedule(instance, "J0-M0:0+3->3,J0-M0:3+4->7,Makespan:7")
        self.assertIn("ambiguous_operation", result.errors)
        self.assertFalse(result.feasible)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.instance = parse_matrix(MATRIX)

    def assert_error(self, text, code):
        result = validate_schedule(self.instance, text)
        self.assertFalse(result.feasible, result)
        self.assertIn(code, result.errors, result)
        self.assertIsNone(result.gap_percent)
        return result

    def test_both_authentic_output_formats(self):
        for text in (SCHEDULE, SHORT_SCHEDULE, "```\n" + SCHEDULE + "```", SCHEDULE.replace("->", "→")):
            with self.subTest(text=text):
                result = validate_schedule(self.instance, text)
                self.assertTrue(result.feasible, result.details)
                self.assertEqual(result.computed_makespan, 7)
                self.assertEqual(result.gap_percent, 0)

    def test_completeness_uniqueness_and_extras(self):
        missing = SCHEDULE.replace("Job 0 Operation 1 on Machine 1 : 3 + 2 -> 5\n", "")
        self.assert_error(missing, "missing_operation")
        self.assert_error(SCHEDULE + "Job 0 Operation 0 on Machine 0 : 0 + 3 -> 3", "duplicate_operation")
        for row in ("Job 8 Operation 0 on Machine 0 : 7+1->8", "Job -1 Operation 0 on Machine 0 : 7+1->8", "J8-M0:7+1->8"):
            self.assert_error(SCHEDULE + row, "extra_operation")

    def test_machine_and_duration_claims_cannot_replace_ground_truth(self):
        self.assert_error(SCHEDULE.replace("Operation 0 on Machine 0", "Operation 0 on Machine 8"), "wrong_machine")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "0 + 2 -> 2"), "wrong_duration")

    def test_arithmetic_and_nonnegative_times(self):
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "0 + 3 -> 4"), "arithmetic_error")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "-1 + 3 -> 2"), "negative_start")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "-0.0000001 + 3 -> 2.9999999"), "negative_start")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "0 + -3 -> -3"), "nonpositive_duration")

    def test_precedence_without_machine_conflict(self):
        # Job 0's second operation precedes its first, but uses the other machine.
        self.assert_error(SCHEDULE.replace("3 + 2 -> 5", "2 + 2 -> 4"), "precedence_violation")

    def test_machine_conflict_without_precedence_violation(self):
        self.assert_error(SCHEDULE.replace("3 + 4 -> 7", "2 + 4 -> 6").replace("Makespan: 7.0", "Makespan: 6.0"), "machine_conflict")

    def test_nested_machine_conflict_is_detected(self):
        instance = parse_matrix("3 1\n0 20\n0 2\n0 1")
        text = "J0-M0:0+20->20,J1-M0:1+2->3,J2-M0:4+1->5,Makespan:20"
        result = validate_schedule(instance, text)
        self.assertEqual(result.errors, ("machine_conflict",))
        self.assertEqual(len(result.details), 2)

    def test_makespan_missing_wrong_duplicate_or_nonfinite(self):
        self.assert_error(SCHEDULE.split("Makespan:")[0], "missing_makespan")
        self.assert_error(SCHEDULE.replace("Makespan: 7.0", "Makespan: 99"), "makespan_mismatch")
        self.assert_error(SCHEDULE + "Makespan:7", "duplicate_makespan")
        self.assert_error(SCHEDULE.replace("Makespan: 7.0", "Makespan: 1e999"), "nonfinite_number")
        self.assertTrue(validate_schedule(self.instance, SCHEDULE.split("Makespan:")[0], require_makespan=False).feasible)

    def test_malformed_extra_rows_are_not_silently_ignored(self):
        for row in ("Job 8 Operation 0 on Machine 0 : nonsense", "J8-M0: NaN+1->1", "J8-M0: 0+1 -> ?"):
            self.assert_error(SCHEDULE + row, "malformed_operation")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "1e999 + 3 -> 1e999"), "nonfinite_number")
        self.assert_error(SCHEDULE.replace("0 + 3 -> 3", "0 + 3 -> 3eBAD"), "malformed_operation")
        self.assert_error(SCHEDULE.replace("Makespan: 7.0", "Makespan: 7eBAD"), "malformed_makespan")
        self.assert_error(SCHEDULE + "Makespan: NaN", "duplicate_makespan")

    def test_empty_output_fails_and_invalid_reference_raises(self):
        result = self.assert_error("", "missing_operation")
        self.assertIsNone(result.computed_makespan)
        for reference in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_schedule(self.instance, SCHEDULE, reference)

    def test_negative_gap_is_allowed_for_nonoptimal_reference(self):
        self.assertAlmostEqual(validate_schedule(self.instance, SCHEDULE, 10).gap_percent, -30)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        instance = parse_matrix(MATRIX)
        self.ok = validate_schedule(instance, SCHEDULE)
        self.better = validate_schedule(instance, SCHEDULE, 10)
        self.bad = validate_schedule(instance, "")

    def test_wilson_known_values_and_zero_denominator(self):
        lo, hi = wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.236593090512564, places=12)
        self.assertAlmostEqual(hi, 0.763406909487436, places=12)
        self.assertEqual(wilson_interval(0, 0), (None, None))
        for args in ((2, 1), (-1, 10), (1, -1), (1.2, 3)):
            with self.assertRaises(ValueError):
                wilson_interval(*args)

    def test_candidate_instance_denominators_and_fixed_best_of_k(self):
        records = [EvaluationRecord("a", 0, self.bad), EvaluationRecord("a", 1, self.ok),
                   EvaluationRecord("b", 0, self.better), EvaluationRecord("b", 1, self.ok),
                   EvaluationRecord("c", 0, self.bad)]
        result = aggregate_results(records, ks=(1, 2, 3))
        self.assertEqual((result["n_candidates"], result["n_instances"]), (5, 3))
        self.assertEqual(result["candidate_feasibility_rate"], 3 / 5)
        self.assertEqual(result["instance_any_feasible_rate"], 2 / 3)
        self.assertEqual(result["conditional_candidate_gap"]["mean_gap_percent"], -10)
        self.assertEqual(result["best_of_k"]["1"]["instance_feasibility_rate"], 1 / 3)
        self.assertEqual(result["best_of_k"]["2"]["n_eligible_instances"], 2)
        self.assertEqual(result["best_of_k"]["2"]["instance_feasibility_rate"], 1)
        self.assertEqual(result["best_of_k"]["2"]["conditional_best_gap"]["mean_gap_percent"], -15)
        self.assertEqual(result["best_of_k"]["3"]["n_eligible_instances"], 0)
        self.assertIsNone(result["best_of_k"]["3"]["instance_feasibility_rate"])
        self.assertEqual(result["error_counts_candidate_denominator"]["missing_operation"], 2)
        self.assertEqual(result["error_rates_instance_denominator"]["missing_operation"], 2 / 3)

    def test_error_instance_count_deduplicates_repeated_candidates(self):
        result = aggregate_results([EvaluationRecord("a", i, self.bad) for i in range(3)])
        self.assertEqual(result["error_counts_candidate_denominator"]["missing_operation"], 3)
        self.assertEqual(result["error_counts_instance_denominator"]["missing_operation"], 1)

    def test_no_feasible_gap_and_no_records_are_null(self):
        result = aggregate_results([EvaluationRecord("a", 0, self.bad)])
        self.assertIsNone(result["conditional_candidate_gap"]["mean_gap_percent"])
        self.assertEqual(result["conditional_candidate_gap"]["n_with_reference"], 0)
        self.assertIsNone(aggregate_results([])["candidate_feasibility_rate"])

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_results([EvaluationRecord("a", 0, self.ok)] * 2)


class PromptSelectionTests(unittest.TestCase):
    def setUp(self):
        instance = parse_matrix(MATRIX)
        self.ok = validate_schedule(instance, SCHEDULE)
        self.better = validate_schedule(instance, SCHEDULE, 10)
        self.bad = validate_schedule(instance, "")

    def test_registry_has_original_three_and_four_unseen(self):
        self.assertEqual([p.id for p in PROMPTS if p.seen_in_training], ["seen_1", "seen_2", "seen_3"])
        self.assertEqual(sum(not p.seen_in_training for p in PROMPTS), 4)
        self.assertTrue(render_user_prompt("seen_1", "INSTANCE").endswith("\n\nINSTANCE"))
        with self.assertRaises(ValueError):
            render_user_prompt("invented_after_test", "x")

    def test_selection_feasibility_then_gap_then_frozen_order(self):
        records = {
            "unseen_1": [EvaluationRecord("v", 0, self.bad)],
            "seen_3": [EvaluationRecord("v", 0, self.better)],
            "seen_2": [EvaluationRecord("v", 0, self.better)],
            "seen_1": [EvaluationRecord("v", 0, self.ok)],
        }
        self.assertEqual(select_validation_prompt(records)["selected_prompt_id"], "seen_2")

    def test_no_test_overlap_or_different_instance_candidate_sets(self):
        records = {"seen_1": [EvaluationRecord("test_hash", 0, self.ok)]}
        with self.assertRaisesRegex(ValueError, "overlaps"):
            select_validation_prompt(records, forbidden_test_instance_ids=["test_hash"])
        with self.assertRaisesRegex(ValueError, "same validation"):
            select_validation_prompt({"seen_1": [EvaluationRecord("v", 0, self.ok)],
                                      "seen_2": [EvaluationRecord("w", 0, self.ok)]})
        with self.assertRaises(ValueError):
            select_validation_prompt({})


if __name__ == "__main__":
    unittest.main()
