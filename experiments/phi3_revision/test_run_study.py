"""CPU-only study-manager tests. No SSH, GPU, training, or paper writes."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

try:
    from .run_study import (ManagerConfig, PROMPTS, SYSTEM_PROMPT, SSHTransport, StudyManager, TransientTransportError, Worker, checkpoint_ready, compile_manuscript,
                            eval_jobs, evaluation_command, evaluation_complete,
                            idle_a6000, json_hash, parse_gpu_inventory, read_study,
                            sha256_file, training_command)
except ImportError:
    from run_study import (ManagerConfig, PROMPTS, SYSTEM_PROMPT, SSHTransport, StudyManager, TransientTransportError, Worker, checkpoint_ready, compile_manuscript,
                           eval_jobs, evaluation_command, evaluation_complete,
                           idle_a6000, json_hash, parse_gpu_inventory, read_study,
                           sha256_file, training_command)


UUID = "GPU-unit-test"
DEVICE = {"index": 0, "uuid": UUID, "name": "NVIDIA RTX A6000", "memory_total_mib": 49140,
          "memory_used_mib": 2, "utilization_percent": 0, "has_compute_process": False}


class FakeTransport:
    def __init__(self, device=None):
        self.device = dict(device or DEVICE)
        self.launches = []

    def inventory(self, host):
        return [dict(self.device)]

    def alive(self, host, pid):
        return True

    def launch(self, host, uuid, command, config, prefix):
        self.launches.append((host, uuid, command, prefix))
        pid = 100 + len(self.launches)
        prefix.with_suffix(".pid").write_text(str(pid))
        return pid


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        study_dir = root / "study"
        study_dir.mkdir()
        for name in ("test.jsonl", "prompt_validation.jsonl", "PROMPTS.json", "PREPARATION_SPLIT.json"):
            (study_dir / name).write_text("{}\n")
        self.study = {
            "artifact_sha256": {path.name: sha256_file(path) for path in study_dir.iterdir()},
            "checkpoint_steps": [0, 461, 922, 1383, 1844], "test_count": 216, "prompt_validation_count": 72,
            "training_budget_comparison_prompt": "seen_1", "prompt_ids": ["seen_1", "seen_2", "seen_3", "unseen_1", "unseen_2", "unseen_3", "unseen_4"],
            "training_recipe": {"optimizer_steps": 1844, "micro_batch_size": 2, "gradient_accumulation_steps": 32,
                                "lora_rank": 128, "lora_alpha": 256, "lora_dropout": .05, "learning_rate": .0002,
                                "warmup_steps": 1, "sequence_cap": 40000, "seed": 42},
            "decoding": {"candidate_count_test": 10, "candidate_count_prompt_validation": 3,
                         "batch_size_test": 5, "batch_size_prompt_validation": 3,
                         "seed": 42, "temperature": 1.0, "top_k": 50, "top_p": .95},
        }
        from dataclasses import asdict
        self.study["prompt_registry_sha256"] = json_hash({"prompts": [asdict(prompt) for prompt in PROMPTS], "system_prompt": SYSTEM_PROMPT})
        self.study["protocol_sha256"] = json_hash(self.study)
        (study_dir / "STUDY.json").write_text(json.dumps(self.study))
        self.config = ManagerConfig(study_dir=str(study_dir), runs_dir=str(root / "runs"),
                                    tokenized_data=str(root / "data" / "train.jsonl"),
                                    validation_data=str(root / "data" / "validation.jsonl"),
                                    training_gpu_uuid=UUID)
        self.transport = FakeTransport()
        self.worker = Worker("worker", "ourasi", 0, UUID)

    def tearDown(self):
        self.tmp.cleanup()

    def manager(self):
        return StudyManager(self.config, transport=self.transport, workers=(self.worker,))

    def checkpoint(self, step):
        directory = Path(self.config.runs_dir) / "main" / f"checkpoint-{step}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text("{}")
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
        (directory / "experiment_checkpoint.json").write_text(json.dumps({"global_step": step}))
        return directory

    def prepared(self):
        directory = Path(self.config.tokenized_data).parent
        directory.mkdir()
        Path(self.config.tokenized_data).write_text("{}\n")
        Path(self.config.validation_data).write_text("{}\n")
        (directory / "manifest.json").write_text(json.dumps({"status": "prepared", "training_count": 118000, "validation_count": 2000}))

    def evaluation(self, name, *, run_kind="FULL", status="complete"):
        directory = Path(self.config.runs_dir) / "evaluations" / name
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"run_kind": run_kind, "expected_candidate_slots": 10}
        manifest["configuration_sha256"] = json_hash(manifest)
        (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
        (directory / "summary.json").write_text(json.dumps({"run_kind": run_kind, "status": status,
            "configuration_sha256": manifest["configuration_sha256"], "terminal_candidate_slots": 10, "not_generated_slots": 0}))
        (directory / "COMPLETE").write_text(json.dumps({"configuration_sha256": manifest["configuration_sha256"], "candidate_slots": 10}))
        (directory / "candidates.jsonl").write_text("{}\n" * 10)
        return directory

    def test_default_preparation_does_not_launch_production(self):
        manager = self.manager()
        self.assertEqual(manager.state["status"], "prepared_not_launched")
        self.assertEqual(self.transport.launches, [])
        self.assertFalse(manager.tick())
        self.assertEqual(self.transport.launches, [])

    def test_frozen_study_tampering_rejected(self):
        (Path(self.config.study_dir) / "test.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            read_study(Path(self.config.study_dir))

    def test_commands_keep_full_training_budget_and_validation_test_budgets(self):
        train = training_command(self.config, self.study)
        self.assertEqual(train[train.index("--total-steps") + 1], "1844")
        self.assertEqual(train[train.index("--micro-batch-size") + 1], "2")
        self.assertEqual(train[train.index("--gradient-accumulation-steps") + 1], "32")
        jobs = eval_jobs(self.config, self.study)
        self.assertNotIn("checkpoint_1844", [job["name"] for job in jobs])
        for job in jobs:
            command = evaluation_command(self.config, self.study, job)
            expected = "3" if job["kind"] == "prompt_validation" else "10"
            self.assertEqual(command[command.index("--candidate-count") + 1], expected)
            self.assertNotIn("--limit", command)
        self.assertEqual(len(jobs), 6)

    def test_no_eval_before_atomic_checkpoint_marker(self):
        manager = self.manager()
        directory = self.checkpoint(0)
        (directory / "experiment_checkpoint.json").unlink()
        manager.tick()
        self.assertEqual(self.transport.launches, [])
        (directory / "experiment_checkpoint.json").write_text('{"global_step":0}')
        manager.tick()
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(manager.state["jobs"]["checkpoint_0"]["status"], "running")

    def test_busy_gpu_is_never_used(self):
        self.prepared()
        self.checkpoint(0)
        self.transport.device["has_compute_process"] = True
        self.manager().tick()
        self.assertEqual(self.transport.launches, [])
        self.assertFalse(idle_a6000(dict(DEVICE, memory_used_mib=400)))
        self.assertFalse(idle_a6000(dict(DEVICE, name="RTX 2080 Ti")))

    def test_uuid_mismatch_aborts(self):
        self.checkpoint(0)
        self.transport.device["uuid"] = "GPU-wrong"
        with self.assertRaisesRegex(ValueError, "UUID changed"):
            self.manager().tick()

    def test_one_job_per_worker_and_state_resume_avoids_duplicate_launch(self):
        self.checkpoint(0)
        self.checkpoint(461)
        manager = self.manager()
        manager.tick()
        manager.tick()
        self.assertEqual(len(self.transport.launches), 1)
        resumed = self.manager()
        resumed.tick()
        self.assertEqual(len(self.transport.launches), 1)
        self.evaluation("checkpoint_0")
        resumed.tick()
        self.assertEqual(len(self.transport.launches), 2)
        self.assertEqual(resumed.state["jobs"]["checkpoint_461"]["status"], "running")

    def test_production_training_launches_once_when_prepared_and_free(self):
        self.prepared()
        manager = self.manager()
        manager.tick()
        manager.tick()
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(manager.state["training"]["status"], "running")

    def test_failed_job_cannot_trigger_report_or_silent_retry(self):
        self.checkpoint(0)
        manager = self.manager()
        manager.tick()
        (manager.directory / "checkpoint_0.exit").write_text("1")
        with patch("subprocess.run") as subprocess_run:
            with self.assertRaisesRegex(RuntimeError, "without a valid completion"):
                manager.tick()
            subprocess_run.assert_not_called()

    def test_prompt_selection_precedes_final_test_and_final_report_is_withheld(self):
        self.checkpoint(1844)
        self.evaluation("final_prompt_validation")
        manager = self.manager()
        def fake_report(command, **kwargs):
            self.assertIn("--select-prompt-only", command)
            (Path(self.config.study_dir) / "selected_prompt.json").write_text('{"selected_prompt_id":"seen_1"}')
            return subprocess.CompletedProcess(command, 0, "selected", "")
        with patch("subprocess.run", side_effect=fake_report) as report:
            manager.tick()
        report.assert_called_once()
        self.assertEqual(manager.state["prompt_selection"], "complete")
        self.assertEqual(manager.state["jobs"]["final_prompt_test"]["status"], "running")
        self.assertEqual(manager.state["report"], "pending")

    def test_pilot_context_skip_and_bad_completion_rejected(self):
        directory = self.evaluation("pilot", run_kind="PILOT_LIMITED")
        with self.assertRaisesRegex(ValueError, "Pilot output"):
            evaluation_complete(directory)
        directory = self.evaluation("skips", status="complete_with_skips")
        with self.assertRaisesRegex(ValueError, "context skips"):
            evaluation_complete(directory)

    def test_inventory_parser_matches_compute_process_by_uuid(self):
        devices = parse_gpu_inventory("0, GPU-test, NVIDIA RTX A6000, 49140, 2, 0\n", "GPU-test, 12345\n")
        self.assertTrue(devices[0]["has_compute_process"])
        self.assertFalse(idle_a6000(devices[0]))
        with self.assertRaises(ValueError):
            parse_gpu_inventory("broken output", "")

    def test_timeout_and_poll_limits(self):
        self.assertEqual(self.config.timeout_days, 0)
        replace(self.config, timeout_days=15).validate()
        for config in (replace(self.config, timeout_days=-1), replace(self.config, poll_seconds=61), replace(self.config, poll_seconds=0)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_compilation_uses_cached_tectonic_and_pypdf_then_preserves_original(self):
        self.manager()
        paper = Path(self.tmp.name) / "paper"
        paper.mkdir()
        (paper / "neurips_2024.pdf").write_bytes(b"old sample PDF")
        config = replace(self.config, paper_dir=str(paper))
        def process(command, **kwargs):
            if command[0] == config.tectonic:
                self.assertEqual(command[1:5], ["-C", "--keep-logs", "--keep-intermediates", "-o"])
                self.assertIn("XDG_CACHE_HOME", kwargs["env"])
                (paper / "build" / "neurips_2024.pdf").write_bytes(b"compiled real manuscript")
                return subprocess.CompletedProcess(command, 0, "compiled", "")
            self.assertEqual(command[0], "/usr/bin/python3")
            self.assertIn("from pypdf import PdfReader", command[2])
            return subprocess.CompletedProcess(command, 0, json.dumps({"pages": 18, "text": "LLMs Can Schedule\nActual results"}), "")
        with patch("subprocess.run", side_effect=process) as process_mock:
            compile_manuscript(config)
        self.assertEqual(process_mock.call_count, 2)
        self.assertEqual((paper / "neurips_2024.pdf").read_bytes(), b"compiled real manuscript")
        self.assertEqual((paper / "neurips_2024.pre_revision.pdf").read_bytes(), b"old sample PDF")

    def test_compilation_never_replaces_pdf_when_sample_checks_fail(self):
        self.manager()
        paper = Path(self.tmp.name) / "paper"
        paper.mkdir()
        (paper / "neurips_2024.pdf").write_bytes(b"preserve me")
        config = replace(self.config, paper_dir=str(paper))
        def process(command, **kwargs):
            if command[0] == config.tectonic:
                (paper / "build" / "neurips_2024.pdf").write_bytes(b"bad compiled PDF")
                return subprocess.CompletedProcess(command, 0, "compiled", "")
            return subprocess.CompletedProcess(command, 0, json.dumps({"pages": 12, "text": "Formatting Instructions for NeurIPS"}), "")
        with patch("subprocess.run", side_effect=process):
            with self.assertRaisesRegex(RuntimeError, "title/page/sample"):
                compile_manuscript(config)
        self.assertEqual((paper / "neurips_2024.pdf").read_bytes(), b"preserve me")
        self.assertFalse((paper / "neurips_2024.pre_revision.pdf").exists())

    def test_transient_inventory_timeout_retries_without_launching(self):
        manager = self.manager()
        with patch.object(manager, "tick", side_effect=[TransientTransportError("temporary SSH timeout"), True]) as tick:
            with patch("time.sleep") as sleep:
                manager.execute()
        self.assertEqual(tick.call_count, 2)
        sleep.assert_called_once_with(60)
        self.assertEqual(self.transport.launches, [])
        self.assertIn("temporary SSH timeout", manager.state["last_transport_error"]["message"])
        manager._lock.close()

    def test_ssh_255_is_transient_and_remote_kill_1_means_dead(self):
        transport = SSHTransport()
        with patch.object(transport, "command", return_value=subprocess.CompletedProcess([], 255, "", "network down")):
            with self.assertRaises(TransientTransportError):
                transport.alive("ourasi", 123)
        with patch.object(transport, "command", return_value=subprocess.CompletedProcess([], 1, "", "no process")):
            self.assertFalse(transport.alive("ourasi", 123))
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 45)):
            with self.assertRaises(TransientTransportError):
                transport.inventory("ourasi")

    def test_status_tracks_training_and_evaluation_progress_without_stale_kill(self):
        manager = self.manager()
        main = Path(self.config.runs_dir) / "main"
        main.mkdir()
        heartbeat = {"global_step": 123, "utc": "old timestamp is informational", "session_training_seconds": 456}
        (main / "training_heartbeat.json").write_text(json.dumps(heartbeat))
        evaluation = Path(self.config.runs_dir) / "evaluations" / "checkpoint_0"
        evaluation.mkdir(parents=True)
        (evaluation / "summary.json").write_text(json.dumps({"terminal_candidate_slots": 25,
            "expected_candidate_slots": 2160, "status": "incomplete", "n_evaluated_instances": 216,
            "by_prompt": {"seen_1": {"feasible_candidates": 7}}}))
        manager.tick()
        self.assertEqual(manager.state["training_progress"], heartbeat)
        self.assertEqual(manager.state["jobs"]["checkpoint_0"]["progress"]["terminal_candidate_slots"], 25)
        self.assertEqual(manager.state["jobs"]["checkpoint_0"]["progress"]["feasible_candidates_by_prompt"], {"seen_1": 7})
        self.assertEqual(manager.state["status"], "running")


if __name__ == "__main__":
    unittest.main()
