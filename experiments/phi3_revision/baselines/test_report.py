"""Focused reporting guards; no model loading, SSH, or manuscript writes."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.phi3_revision.core import EvaluationRecord, ValidationResult
from experiments.phi3_revision.baselines import report


def record(identity, candidate, gap):
    valid = gap is not None
    result = ValidationResult(valid, () if valid else ("missing_operation",), (), (),
                              100 if valid else None, 100 if valid else None, gap)
    return EvaluationRecord(identity, candidate, result)


class GuardTests(unittest.TestCase):
    def test_self_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            value = {"status": "complete"}
            value["receipt_sha256"] = report._hash(value)
            path.write_text(json.dumps(value))
            self.assertEqual(report._self_hashed(path, "receipt_sha256"), value)
            value["status"] = "partial"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Invalid receipt_sha256"):
                report._self_hashed(path, "receipt_sha256")

    def test_incomplete_requested_production_fails_before_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "production is incomplete"):
                report._verify_completed_artifacts(temporary, temporary)

    def test_receipt_requires_nonpilot_216_by_10_by_four(self):
        protocol = {"protocol_sha256": "frozen"}
        receipt = {"status": "complete", "purpose": report.PURPOSE, "protocol_sha256": "frozen",
                   "instances": 216, "candidates_per_instance": 10,
                   "models": {name: {"candidates": 2160, "raw_file_sha256": {str(i): "x" for i in range(216)}} for name in report.MODELS}}
        report._receipt_header(receipt, protocol)
        for key, value in (("purpose", "EXCLUDED_TECHNICAL_PILOT"), ("instances", 215),
                           ("candidates_per_instance", 9), ("status", "incomplete")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                report._receipt_header(dict(receipt, **{key: value}), protocol)
        broken = json.loads(json.dumps(receipt))
        broken["models"][report.MODELS[0]]["candidates"] = 2159
        with self.assertRaisesRegex(ValueError, "lacks all candidates"):
            report._receipt_header(broken, protocol)

    def test_population_rejects_missing_duplicate_and_infeasible_gap(self):
        identities = ["a", "b"]
        records = [record(identity, candidate, 5.0) for identity in identities for candidate in range(10)]
        metric = report._population(records, identities, "test")
        self.assertEqual(metric["n_candidates"], 20)
        for invalid in (records[:-1], records + [records[0]], [replace(records[0], instance_id="outside")] + records[1:]):
            with self.assertRaises(ValueError):
                report._population(invalid, identities, "test")
        inconsistent = replace(records[0], validation=replace(records[0].validation, feasible=False))
        with self.assertRaisesRegex(ValueError, "infeasible candidate has"):
            report._population([inconsistent] + records[1:], identities, "test")

    def test_observed_maximum_uses_each_instances_best_valid_candidate(self):
        records = [record("a", candidate, gap) for candidate, gap in enumerate([5, 70] + [None] * 8)]
        records += [record("b", candidate, gap) for candidate, gap in enumerate([6, 8] + [None] * 8)]
        metric = report._population(records, ["a", "b"], "test")
        self.assertEqual(metric["observed_max_best_of_10_gap_percent"], 6)

    def test_observed_maximum_is_null_without_feasibility_and_defined_for_one_instance(self):
        records = [record("a", candidate, None) for candidate in range(10)]
        self.assertIsNone(report._population(records, ["a"], "test")["observed_max_best_of_10_gap_percent"])
        records[3] = record("a", 3, -2)
        self.assertEqual(report._population(records, ["a"], "test")["observed_max_best_of_10_gap_percent"], -2)

    def test_pairing_uses_identical_phi_feasible_subset_and_best_candidates(self):
        phi = [record("a", 0, 20), record("a", 1, 10), record("b", 0, None), record("c", 0, 30)]
        baseline = [record("a", 0, 5), record("b", 0, 99), record("c", 0, 15)]
        paired = report._paired_gap_summary(phi, baseline)
        self.assertEqual(paired["n_phi_feasible_instances"], 2)
        self.assertEqual(paired["n_paired_instances"], 2)
        self.assertEqual(paired["phi_gap"]["mean_percent"], 20)
        self.assertEqual(paired["baseline_gap"]["mean_percent"], 10)
        self.assertEqual(paired["baseline_minus_phi_gap_percentage_points"]["mean_percent"], -10)

    def test_pairing_does_not_silently_remove_baseline_failure(self):
        paired = report._paired_gap_summary([record("a", 0, 10), record("b", 0, 20)],
                                            [record("a", 0, 5), record("b", 0, None)])
        self.assertEqual(paired["n_phi_feasible_instances"], 2)
        self.assertFalse(paired["subset_complete"])
        self.assertEqual(paired["n_baseline_infeasible_on_phi_subset"], 1)
        self.assertIsNone(paired["baseline_gap"]["mean_percent"])
        self.assertEqual(paired["n_paired_instances"], 0)

    def test_empty_phi_feasible_subset_has_null_gaps(self):
        paired = report._paired_gap_summary([record("a", 0, None)], [record("a", 0, 5)])
        self.assertEqual(paired["n_phi_feasible_instances"], 0)
        self.assertIsNone(paired["baseline_gap"]["mean_percent"])

    def test_legacy_audit_failure_and_wrong_certificate_fail_closed(self):
        protocol = {"protocol_sha256": "frozen", "provenance": {"dependencies": {"hostname": "seacove"}}}
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "bad legacy state")):
            with self.assertRaisesRegex(ValueError, "legacy state"):
                report._legacy_audit("/study", "/baseline", protocol, legacy_python="/legacy/bin/python")
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"status":"verified","protocol_sha256":"other"}', "")):
            with self.assertRaisesRegex(ValueError, "does not match"):
                report._legacy_audit("/study", "/baseline", protocol, legacy_python="/legacy/bin/python")

    def test_tex_keeps_full_and_paired_population_labels_and_valid_row_endings(self):
        records = [record("a", candidate, 5.0) for candidate in range(10)]
        metric = report._population(records, ["a"], "test")
        paired = {"seen_1": {"slj_b32": report._paired_gap_summary(records, records)}}
        tex = report._tex({"slj_b32": metric}, {"seen_1": metric}, paired, "seen_1")
        self.assertIn("different feasible subsets do not establish overall superiority", tex)
        self.assertIn("identical subset", tex)
        self.assertIn("frozen BatchNorm statistics", tex)
        self.assertIn("Observed max gap", tex)
        self.assertIn("not a worst-case guarantee", tex)
        data_rows = [line for line in tex.splitlines() if line.startswith(("SLJ B32 &", "Phi-3 seen", "seen\\_1 &"))]
        self.assertEqual(len(data_rows), 3)
        self.assertTrue(all(line.endswith(r" \\") for line in data_rows))


if __name__ == "__main__":
    unittest.main()
