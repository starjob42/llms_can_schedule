"""Focused SYNTHETIC orchestration fixtures; no model or actual paper writes."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

try:
    from . import journal_finish as finish
except ImportError:
    import journal_finish as finish


def configuration(base):
    return finish.Config(paper_dir=base / "paper", study_dir=base / "study",
                         runs_dir=base / "runs", auto_protocol=base / "auto/PROTOCOL.json",
                         publisher_receipt=base / "publisher/COMPLETE.json",
                         output_dir=base / "finisher", poll_seconds=.01).normalized()


def synthetic_publisher(config, *, complete=True, exit_code=0, status="complete"):
    """The actual publisher field names, with explicitly synthetic evidence."""
    evidence = config.publisher_receipt.parent / "SYNTHETIC_attempt"
    evidence.mkdir(parents=True, exist_ok=True)
    if complete:
        value = finish.hashed({"status": "complete", "purpose": "VERIFIED_AUTO_PROMPT_PUBLICATION",
                               "protocol_sha256": finish.bridge.AUTO_PROTOCOL,
                               "evidence_dir": str(evidence), "final_root_audit_required": True})
        finish.write_json(config.publisher_receipt, value)
        shutil.copy2(config.publisher_receipt, evidence / "COMPLETE.json")
    if status:
        finish.write_json(config.publisher_receipt.parent / "STATUS.json",
                          {"status": status, "stage": "atomic_submission_pdf_replacement",
                           "evidence_dir": str(evidence), "pid": 12345})
    if exit_code is not None:
        finish.write_json(config.publisher_receipt.parent / "EXIT.json",
                          {"exit_code": exit_code, "at_utc": finish.now(), "evidence_dir": str(evidence)})


def transaction_fixture(config):
    config.paper_dir.mkdir()
    attempt = config.output_dir / "SYNTHETIC_transaction"
    attempt.mkdir(parents=True)
    before = {}
    expected = {}
    for name in finish.BRIDGE_TARGETS:
        path = config.paper_dir / name
        path.write_text("SYNTHETIC original " + name)
        before[name] = path.read_bytes()
        expected[str(path)] = __import__("hashlib").sha256(("SYNTHETIC rendered " + name).encode()).hexdigest()
    saved = finish.preserve_files([config.paper_dir / name for name in finish.BRIDGE_TARGETS], attempt / "backups")
    value = finish.hashed({"paper_dir": str(config.paper_dir), "publisher_receipt_file_sha256": finish.digest(config.publisher_receipt),
                           "before": saved, "expected_sha256": expected})
    path = attempt / "BRIDGE_TRANSACTION.json"
    finish.write_json(path, value)
    return path, before


class PublisherAndGuardTests(unittest.TestCase):
    def test_default_readiness_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_journal_finish_") as directory:
            base = Path(directory)
            result = finish.readiness(configuration(base))
            self.assertEqual(result["status"], "waiting_frozen_publication")
            self.assertFalse(result["full_publication_guards_checked"])
            self.assertEqual(list(base.iterdir()), [])

    def test_publisher_requires_successful_matching_tail(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_publisher_tail_") as directory:
            config = configuration(Path(directory))
            synthetic_publisher(config, status="running", exit_code=None)
            self.assertEqual(finish.observe_publisher(config)["status"], "waiting_publisher_tail")
            synthetic_publisher(config)
            self.assertEqual(finish.observe_publisher(config)["status"], "publisher_settled")
            synthetic_publisher(config, status="failed", exit_code=1)
            with self.assertRaisesRegex(finish.FinishError, "completion tail"):
                finish.observe_publisher(config)
            synthetic_publisher(config, status="running", exit_code=1)
            with self.assertRaisesRegex(finish.FinishError, "completion tail"):
                finish.observe_publisher(config)

    def test_settled_publisher_does_not_bypass_actual_missing_training_guard(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_missing_training_") as directory:
            config = configuration(Path(directory))
            config.paper_dir.mkdir()
            (config.paper_dir / "rebuttal.tex").write_text("SYNTHETIC original response")
            synthetic_publisher(config)
            with patch.object(finish.bridge, "integrate_verified_results") as integrate:
                with self.assertRaisesRegex(finish.bridge.PublicationGuardError, "training_complete"):
                    finish.execute(config)
                integrate.assert_not_called()
            self.assertEqual((config.paper_dir / "rebuttal.tex").read_text(), "SYNTHETIC original response")
            self.assertFalse((config.paper_dir / "build").exists())
            failure = finish.read_json(config.output_dir / "EXIT.json")
            self.assertEqual(failure["exit_code"], 1)
            self.assertEqual(failure["stage"], "verify_completed_publication_chain")

    def test_actual_manager_failure_states_stop_waiting(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_manager_failure_") as directory:
            config = configuration(Path(directory))
            for status in ("failed_incomplete", "interrupted_incomplete", "timeout_incomplete"):
                finish.write_json(config.runs_dir / "manager/status.json", {"status": status})
                with self.assertRaisesRegex(finish.FinishError, status):
                    finish.observe_publisher(config)

    def test_actual_bridge_rejects_pilot_without_publication_writes(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_pilot_guard_") as directory:
            config = configuration(Path(directory))
            study = finish.hashed({"purpose": "SYNTHETIC_TEST_ONLY"}, "protocol_sha256")
            finish.write_json(config.study_dir / "STUDY.json", study)
            protocol = finish.hashed({"purpose": "SYNTHETIC_TEST_ONLY", "output_dir": str(Path(directory) / "scientific"),
                "parent_protocol_sha256": study["protocol_sha256"], "parent_study": str(config.study_dir / "STUDY.json"),
                "runs_dir": str(config.runs_dir), "training_complete": str(config.runs_dir / "main/training_complete.json")}, "protocol_sha256")
            finish.write_json(config.auto_protocol, protocol)
            finish.write_json(config.runs_dir / "main/training_complete.json", {"pilot": True, "global_step": 1844})
            finish.write_json(config.runs_dir / "manager/COMPLETE", {})
            for name in ("COMPLETE.json", "INTEGRATED.json"):
                finish.write_json(Path(protocol["output_dir"]) / name, finish.hashed({"purpose": "SYNTHETIC_TEST_ONLY"}))
            with patch.object(finish.bridge, "MAIN_PROTOCOL", study["protocol_sha256"]), patch.object(finish.bridge, "AUTO_PROTOCOL", protocol["protocol_sha256"]):
                synthetic_publisher(config)
                with self.assertRaisesRegex(finish.bridge.PublicationGuardError, "Full non-pilot training"):
                    finish.execute(config)
            self.assertFalse(config.paper_dir.exists())


class RecoveryTests(unittest.TestCase):
    def test_interruption_after_rebuttal_write_restores_all_known_bytes(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_bridge_crash_") as directory:
            config = configuration(Path(directory))
            synthetic_publisher(config)
            transaction, before = transaction_fixture(config)
            for name in finish.BRIDGE_TARGETS:
                (config.paper_dir / name).write_text("SYNTHETIC rendered " + name)
            result = finish.recover_uncommitted_bridge(config, transaction)
            self.assertEqual(len(result["restored"]), 6)
            self.assertEqual({name: (config.paper_dir / name).read_bytes() for name in before}, before)
            self.assertEqual(len(result["preserved_partial_outputs"]), 6)
            self.assertFalse((config.paper_dir / "build" / finish.bridge.RECEIPT).exists())

    def test_external_edit_prevents_any_rollback(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_external_edit_") as directory:
            config = configuration(Path(directory))
            synthetic_publisher(config)
            transaction, before = transaction_fixture(config)
            (config.paper_dir / "rebuttal.tex").write_text("SYNTHETIC rendered rebuttal.tex")
            (config.paper_dir / "journal_intro.tex").write_text("SYNTHETIC external author edit")
            snapshot = {name: (config.paper_dir / name).read_bytes() for name in before}
            with self.assertRaisesRegex(finish.FinishError, "Unexpected paper edit"):
                finish.recover_uncommitted_bridge(config, transaction)
            self.assertEqual(snapshot, {name: (config.paper_dir / name).read_bytes() for name in before})

    def test_builder_child_keeps_lock_after_parent_descriptor_closes(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_inherited_lock_") as directory:
            path = Path(directory)
            child = None
            try:
                with finish.execution_lock(path) as descriptor:
                    child = subprocess.Popen([sys.executable, "-S", "-B", "-c",
                        "import sys;print('ready',flush=True);sys.stdin.readline()"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, pass_fds=(descriptor,))
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaisesRegex(finish.FinishError, "surviving builder child"):
                    with finish.execution_lock(path):
                        self.fail("lock incorrectly released")
            finally:
                if child:
                    child.communicate("release\n", timeout=10)
            with finish.execution_lock(path):
                pass


class CompletionTests(unittest.TestCase):
    def test_actual_final_package_guard_accepts_bound_records_and_rejects_missing_raw(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_final_archive_guard_") as directory:
            root = Path(directory)
            revision = root / "experiments/phi3_revision"
            config = finish.Config(paper_dir=root / "paper", study_dir=revision / "study",
                runs_dir=revision / "runs", auto_protocol=revision / "auto/PROTOCOL.json",
                publisher_receipt=revision / "publisher/COMPLETE.json", output_dir=root / "finisher").normalized()
            paper, native, author = config.paper_dir, root / "SYNTHETIC_native", root / "author"
            paper.mkdir(); native.mkdir()
            sources = ["manuscript.tex", "journal_title_acknowledgments.tex"]
            for name in sources + ["rebuttal.tex"]:
                (paper / name).write_text("SYNTHETIC source: " + name)
            finish.write_json(config.publisher_receipt, {"purpose": "SYNTHETIC_TEST_ONLY"})
            bridge_path = paper / "build" / finish.bridge.RECEIPT
            finish.write_json(bridge_path, {"purpose": "SYNTHETIC_TEST_ONLY"})
            finish.write_json(author / "AUTHOR_DECLARATIONS.json", {"purpose": "SYNTHETIC_TEST_ONLY"})
            finish.write_json(config.auto_protocol, finish.hashed({"output_dir": str(revision / "auto/production")}, "protocol_sha256"))
            required = ["runs/main/training_complete.json", "runs/main/run_manifest.json", "runs/main/training_arguments.json",
                        "study/integrated_results.json", "study/selected_prompt.json"]
            raw = revision / "runs/evaluations/checkpoint_0/candidates.jsonl"
            for name in required:
                finish.write_json(revision / name, {"purpose": "SYNTHETIC_TEST_ONLY"})
            raw.parent.mkdir(parents=True); raw.write_text('{"purpose":"SYNTHETIC_TEST_ONLY"}\n')
            pdf_checks = {}
            for name in ("manuscript", "rebuttal"):
                (native / (name + ".pdf")).write_bytes(b"SYNTHETIC hash-only PDF fixture; not an actual PDF")
                (native / (name + ".log")).write_text("SYNTHETIC successful-log fixture")
                shutil.copy2(native / (name + ".pdf"), paper / (name + ".pdf"))
                pdf_checks[name] = {"sha256": finish.digest(native / (name + ".pdf")), "pages": 1,
                    "undefined_references_or_glyphs": False, "overfull_boxes": False,
                    "final_log_sha256": finish.digest(native / (name + ".log"))}
            (native / "manuscript.bbl").write_text("SYNTHETIC bibliography fixture")
            source_zip = paper / "LLMs_can_Schedule_journal_sources.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                for name in sources:
                    archive.write(paper / name, name)
                archive.write(native / "manuscript.pdf", "manuscript.pdf")
                archive.write(native / "manuscript.bbl", "manuscript.bbl")
                archive.writestr("README.txt", "SYNTHETIC_TEST_ONLY")
            supplement = paper / "ESM_1.zip"
            def make_supplement(include_raw):
                paths = [revision / name for name in required] + ([raw] if include_raw else [])
                manifest = finish.hashed({"contains_final_results": True, "frozen_scientific_bytes_rewritten": False,
                    "file_sha256": {path.relative_to(root).as_posix(): finish.digest(path) for path in paths}}, "manifest_sha256")
                with zipfile.ZipFile(supplement, "w") as archive:
                    for path in paths:
                        archive.write(path, path.relative_to(root).as_posix())
                    archive.writestr("MANIFEST.json", json.dumps(manifest))
                return {"path": str(supplement), "sha256": finish.digest(supplement),
                    "contains_final_results": True, "archive_manifest_sha256": manifest["manifest_sha256"]}
            receipt = {"status": "compiled_and_checked", "experimental_results_pending": False,
                "upstream_journal_integration_receipt_sha256": finish.digest(bridge_path),
                "frozen_publisher_receipt_file_sha256": finish.digest(config.publisher_receipt),
                "source_sha256": {name: finish.digest(paper / name) for name in sources},
                "rebuttal_source_sha256": finish.digest(paper / "rebuttal.tex"),
                "author_metadata": {"author_declarations_sha256": finish.digest(author / "AUTHOR_DECLARATIONS.json"),
                                   "contribution_roles_supplied": True},
                "compiler": {"status": "compiled", "stages": [{"returncode": 0}] * 7},
                "build_dir": str(native), "pdf_checks": pdf_checks,
                "source_package": {"path": str(source_zip), "sha256": finish.digest(source_zip)},
                "supplementary": make_supplement(True)}
            finish.write_json(paper / "JOURNAL_BUILD.json", finish.hashed(receipt))
            verified = {"bindings": {str(raw): finish.digest(raw)}}
            with patch.object(finish, "ROOT", root), patch.object(finish, "HERE", revision), \
                 patch.object(finish.builder, "SOURCES", sources), patch.object(finish.builder, "ART", author):
                value = finish.verify_final_build(config, verified)
                self.assertFalse(value["final_root_review_complete"])
                self.assertTrue(value["author_contributions_supplied"])
                # Rebind every outer hash as a valid-but-incomplete archive:
                # coverage must reject the missing raw member independently.
                receipt["supplementary"] = make_supplement(False)
                finish.write_json(paper / "JOURNAL_BUILD.json", finish.hashed(receipt))
                with self.assertRaisesRegex(finish.FinishError, "omits/changes verified scientific record"):
                    finish.verify_final_build(config, verified)

    def test_pending_build_is_never_completion(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_pending_build_") as directory:
            config = configuration(Path(directory))
            finish.write_json(config.paper_dir / "JOURNAL_BUILD.json", finish.hashed(
                {"status": "compiled_and_checked", "experimental_results_pending": True}))
            with self.assertRaisesRegex(finish.FinishError, "pending or incomplete"):
                finish.verify_final_build(config, {})

    def test_complete_reuse_never_restarts_builder_or_bridge(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_idempotence_") as directory:
            config = configuration(Path(directory))
            synthetic_publisher(config)
            finish.write_json(config.paper_dir / "JOURNAL_BUILD.json", {"experimental_results_pending": False})
            finish.write_json(config.paper_dir / "build" / finish.bridge.RECEIPT, {"purpose": "SYNTHETIC_TEST_ONLY"})
            build = Mock(side_effect=AssertionError("builder must not restart"))
            with patch.object(finish, "verify_chain", return_value={"prior": {"purpose": "SYNTHETIC_TEST_ONLY"}}), \
                 patch.object(finish, "verify_final_build", return_value={"purpose": "SYNTHETIC_TEST_ONLY", "final_root_review_complete": False}), \
                 patch.object(finish.bridge, "integrate_verified_results") as integrate:
                first = finish.execute(config, build=build)
                second = finish.execute(config, build=build)
            self.assertEqual(first["status"], "awaiting_final_root_review")
            self.assertEqual(second["status"], "awaiting_final_root_review")
            self.assertFalse(second["scientific_approval_asserted"])
            self.assertFalse(second["submitted_or_published_to_journal"])
            self.assertNotEqual(first["attempt_dir"], second["attempt_dir"])
            self.assertTrue((Path(first["attempt_dir"]) / "COMPLETE.json").is_file())
            build.assert_not_called()
            integrate.assert_not_called()

    def test_changed_build_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_build_drift_") as directory:
            base = Path(directory)
            body = base / "journal_body.tex"
            body.write_text("SYNTHETIC original methods")
            snapshot = base / "BUILD_INPUTS.json"
            finish.write_json(snapshot, finish.hashed({"stable_sha256": {str(body): finish.digest(body)},
                              "generated_author_sha256": {}, "original_outputs": {}}))
            body.write_text("SYNTHETIC concurrent external edit")
            with self.assertRaisesRegex(finish.FinishError, "changed during build"):
                finish.check_build_inputs(snapshot, base, generated_required=True)

    def test_resume_cannot_accept_final_receipt_after_during_compile_source_drift(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_final_resume_drift_") as directory:
            config = configuration(Path(directory))
            synthetic_publisher(config)
            finish.write_json(config.paper_dir / "JOURNAL_BUILD.json", {"experimental_results_pending": False})
            body = config.paper_dir / "journal_body.tex"
            body.write_text("SYNTHETIC before compile")
            snapshot = config.output_dir / "SYNTHETIC_previous/BUILD_INPUTS.json"
            finish.write_json(snapshot, finish.hashed({"stable_sha256": {str(body): finish.digest(body)},
                              "generated_author_sha256": {}, "original_outputs": {}}))
            finish.write_json(config.output_dir / "STATUS.json", {"build_input_snapshot": str(snapshot)})
            finish.write_json(config.output_dir / "COMPLETE.json", {"purpose": "SYNTHETIC previous current marker"})
            body.write_text("SYNTHETIC edit while old builder compiled")
            with patch.object(finish, "verify_chain", return_value={"prior": {"purpose": "SYNTHETIC_TEST_ONLY"}}), \
                 patch.object(finish, "verify_final_build") as verify_final:
                with self.assertRaisesRegex(finish.FinishError, "changed during build"):
                    finish.execute(config)
                verify_final.assert_not_called()
            state = finish.read_json(config.output_dir / "STATUS.json")
            self.assertEqual(state["build_input_snapshot"], str(snapshot))
            self.assertEqual(state["status"], "failed")
            self.assertFalse((config.output_dir / "COMPLETE.json").exists())
            self.assertTrue((Path(state["attempt_dir"]) / "PRIOR_CURRENT_COMPLETE.json").exists())

    def test_poll_interval_is_finite(self):
        for value in (0, -1, 61, float("inf")):
            with self.assertRaises(finish.FinishError):
                replace(finish.Config(), poll_seconds=value).normalized()


if __name__ == "__main__":
    unittest.main()
