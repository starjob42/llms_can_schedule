"""CPU-only evaluation orchestration tests with a deterministic fake generator."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .evaluate import (EVALUATION_SYSTEM_PROMPT, EvaluationConfig, build_manifest, eos_token_ids, evaluation_user_prompt, generation_seed,
                           load_dataset, load_progress, run, token_budget)
    from .evaluate import trim_generated_tokens
except ImportError:
    from evaluate import (EVALUATION_SYSTEM_PROMPT, EvaluationConfig, build_manifest, eos_token_ids, evaluation_user_prompt, generation_seed,
                          load_dataset, load_progress, run, token_budget)
    from evaluate import trim_generated_tokens


MATRIX = "2 2\n0 3 1 2\n1 2 0 4\n7"
TEXT = """Optimize schedule for 2 Jobs across 2 Machines to minimize makespan.
Machine 0 is used for the following Operations:
Job 0 Operation 0 duration 3 mins.
Job 1 Operation 1 duration 4 mins.
Machine 1 is used for the following Operations:
Job 0 Operation 1 duration 2 mins.
Job 1 Operation 0 duration 2 mins.
"""
OUTPUT = "J0-M0:0+3->3,J1-M1:0+2->2,J0-M1:3+2->5,J1-M0:3+4->7,Makespan:7"


class FakeEncoded:
    shape = (1, 100)


class FakeEngine:
    max_context = 10000

    def __init__(self, *, fail_after=None, output=OUTPUT):
        self.calls = []
        self.fail_after = fail_after
        self.output = output

    def encode(self, user_text):
        return FakeEncoded()

    def generate(self, encoded, count, seed, max_new_tokens):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("CUDA out of memory (simulated)")
        self.calls.append((count, seed, max_new_tokens))
        return [{"output": self.output, "input_tokens": 100, "output_tokens": 80,
                 "batch_seconds": 2.0, "amortized_seconds": 2.0 / count,
                 "hit_max_new_tokens": False} for _ in range(count)]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"fake weights; no GPU/model is used")
        (self.base / "config.json").write_text('{"model_type":"phi3"}')
        self.data = self.root / "data.json"
        self.row = {"matrix": MATRIX, "prompt_machines_first": TEXT,
                    "prompt_jobs_first": TEXT, "output": OUTPUT}
        self.data.write_text(json.dumps([self.row]))
        self.config = EvaluationConfig(str(self.base), str(self.data), str(self.root / "result"),
                                       prompt_ids=("seen_1", "unseen_1"), candidate_count=3, batch_size=2)

    def tearDown(self):
        self.tmp.cleanup()

    def raw_rows(self):
        return [json.loads(line) for line in (Path(self.config.output_dir) / "candidates.jsonl").read_text().splitlines()]

    def test_full_fixed_population_and_paired_seeds(self):
        engine = FakeEngine()
        summary = run(self.config, engine=engine)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["terminal_candidate_slots"], 6)
        self.assertEqual(summary["not_generated_slots"], 0)
        self.assertTrue((Path(self.config.output_dir) / "COMPLETE").exists())
        self.assertEqual([call[0] for call in engine.calls], [2, 1, 2, 1])
        self.assertEqual(engine.calls[:2], engine.calls[2:])
        for result in summary["by_prompt"].values():
            self.assertEqual(result["candidate_feasibility_rate"], 1)
            self.assertEqual(result["conditional_candidate_gap"]["mean_gap_percent"], 0)
        for row in self.raw_rows():
            self.assertEqual(row["input_tokens"], 100)
            self.assertEqual(row["max_new_tokens"], 4 * 48 + 128)
            self.assertIn("generation_seed", row)

    def test_invalid_candidates_do_not_trigger_early_stopping(self):
        engine = FakeEngine(output="no usable schedule")
        summary = run(self.config, engine=engine)
        self.assertEqual(summary["terminal_candidate_slots"], 6)
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual(summary["by_prompt"]["seen_1"]["candidate_feasibility_rate"], 0)

    def test_complete_resume_revalidates_outputs_without_inference(self):
        run(self.config, engine=FakeEngine())
        progress = Path(self.config.output_dir) / "candidates.jsonl"
        rows = self.raw_rows()
        # A corrupt cached feasibility claim must not affect recomputation.
        rows[0]["validation"]["feasible"] = False
        progress.write_text("".join(json.dumps(row) + "\n" for row in rows))
        engine = FakeEngine(fail_after=0)
        summary = run(self.config, engine=engine)
        self.assertEqual(engine.calls, [])
        self.assertEqual(summary["by_prompt"]["seen_1"]["feasible_candidates"], 3)

    def test_oom_preserves_progress_and_resume_completes(self):
        engine = FakeEngine(fail_after=1)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            run(self.config, engine=engine)
        directory = Path(self.config.output_dir)
        self.assertFalse((directory / "COMPLETE").exists())
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["terminal_candidate_slots"], 2)
        self.assertEqual(summary["not_generated_slots"], 4)
        # The entire planned candidate denominator remains visible after OOM.
        self.assertEqual(summary["by_prompt"]["seen_1"]["n_candidates"], 3)
        self.assertEqual(summary["by_prompt"]["unseen_1"]["n_candidates"], 3)
        failure = json.loads((directory / "failures.jsonl").read_text())
        self.assertTrue(failure["is_cuda_oom"])
        resumed = run(self.config, engine=FakeEngine())
        self.assertEqual(resumed["terminal_candidate_slots"], 6)

    def test_partial_batch_regenerates_same_seed_and_skips_persisted_rows(self):
        run(self.config, engine=FakeEngine())
        directory = Path(self.config.output_dir)
        first = self.raw_rows()[0]
        (directory / "candidates.jsonl").write_text(json.dumps(first) + "\n")
        engine = FakeEngine()
        run(self.config, engine=engine)
        self.assertEqual(engine.calls[0][1], first["generation_seed"])
        self.assertEqual(len(self.raw_rows()), 6)

    def test_changed_configuration_data_or_weights_refuses_resume(self):
        run(self.config, engine=FakeEngine())
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run(replace(self.config, seed=43), engine=FakeEngine())
        self.data.write_text(json.dumps([self.row], indent=2))
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run(self.config, engine=FakeEngine())
        self.data.write_text(json.dumps([self.row]))
        (self.base / "model.safetensors").write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run(self.config, engine=FakeEngine())

    def test_context_skips_are_population_failures(self):
        engine = FakeEngine()
        engine.max_context = 110
        summary = run(self.config, engine=engine)
        self.assertEqual(summary["status"], "complete_with_skips")
        self.assertEqual(summary["candidate_status_counts"]["skipped_context"], 6)
        self.assertEqual(engine.calls, [])
        self.assertEqual(summary["by_prompt"]["seen_1"]["candidate_feasibility_rate"], 0)
        self.assertEqual(summary["by_prompt"]["seen_1"]["error_counts_candidate_denominator"]["skipped_context"], 3)

    def test_pilots_are_explicitly_stamped(self):
        config = replace(self.config, limit=1)
        summary = run(config, engine=FakeEngine())
        manifest = json.loads((Path(config.output_dir) / "evaluation_manifest.json").read_text())
        self.assertEqual(summary["run_kind"], "PILOT_LIMITED")
        self.assertEqual(manifest["run_kind"], "PILOT_LIMITED")

    def test_duplicate_canonical_instances_fail_even_outside_pilot_limit(self):
        self.data.write_text(json.dumps([self.row, dict(self.row, matrix=MATRIX.replace("\n7", "\n99"), output=OUTPUT)]))
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            load_dataset(self.data, limit=1)

    def test_matrix_fallback_and_jsonl(self):
        row = dict(self.row)
        row.pop("matrix")
        path = self.root / "data.jsonl"
        path.write_text(json.dumps(row) + "\n")
        items, population = load_dataset(path)
        self.assertEqual(population, 1)
        self.assertEqual(items[0].instance.reference_makespan, 7)

    def test_compact_machine_input_audited_and_marked_missing_precedence(self):
        row = dict(self.row, prompt_machines_first="M0:\nJ0:3 J1:4\nM1:\nJ1:2 J0:2")
        self.data.write_text(json.dumps([row]))
        items, _ = load_dataset(self.data)
        self.assertFalse(items[0].input_encodes_precedence)
        self.data.write_text(json.dumps([dict(row, prompt_machines_first=row["prompt_machines_first"].replace("J0:3", "J0:99"))]))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_dataset(self.data)

    def test_corrupt_progress_duplicate_wrong_seed_and_wrong_config_rejected(self):
        run(self.config, engine=FakeEngine())
        items, _ = load_dataset(self.data)
        path = Path(self.config.output_dir) / "candidates.jsonl"
        first = self.raw_rows()[0]
        path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate persisted"):
            load_progress(path, items, self.config)
        first["generation_seed"] += 1
        path.write_text(json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "wrong paired seed"):
            load_progress(path, items, self.config)
        with self.assertRaisesRegex(ValueError, "another configuration"):
            load_progress(path, items, self.config, "wrong configuration")
        path.write_text('{"partial":')
        with self.assertRaisesRegex(ValueError, "Corrupt candidate"):
            load_progress(path, items, self.config)

    def test_bad_labels_and_prompt_matrix_mismatch_fail_before_loading(self):
        self.data.write_text(json.dumps([dict(self.row, output="Makespan:7")]))
        with self.assertRaisesRegex(ValueError, "Invalid solver label"):
            run(self.config, engine=FakeEngine())
        self.data.write_text(json.dumps([dict(self.row, prompt_machines_first=TEXT.replace("duration 3", "duration 5"))]))
        with self.assertRaisesRegex(ValueError, "Input/matrix mismatch"):
            run(self.config, engine=FakeEngine())

    def test_seed_uses_instance_and_batch_but_not_checkpoint_or_prompt(self):
        self.assertEqual(generation_seed("instance", 42, 0), generation_seed("instance", 42, 0))
        self.assertNotEqual(generation_seed("instance", 42, 0), generation_seed("instance", 42, 1))
        self.assertNotEqual(generation_seed("instance", 42, 0), generation_seed("different", 42, 0))

    def test_original_release_prompt_and_sampling_defaults(self):
        self.assertEqual(EVALUATION_SYSTEM_PROMPT, "You are an expert in Job Shop Scheduling Problem")
        expected = "Instruct: Provide a solution schedule for the JSSP problem below, also indicate the makespan.\nINSTANCE"
        self.assertEqual(evaluation_user_prompt("seen_1", "INSTANCE"), expected)
        self.assertEqual((self.config.temperature, self.config.top_k, self.config.top_p), (1.0, 50, 0.95))

    def test_all_official_phi3_stop_tokens_are_honored(self):
        stops = eos_token_ids([32000, 32001, 32007], 32000)
        self.assertEqual(stops, (32000, 32001, 32007))
        self.assertEqual(trim_generated_tokens([1, 2, 32007, 32000, 32000], stops), [1, 2, 32007])
        self.assertEqual(trim_generated_tokens([1, 32001, 5], stops), [1, 32001])
        self.assertEqual(trim_generated_tokens([1, 2], stops), [1, 2])
        self.assertEqual(eos_token_ids(None, 32000), (32000,))
        for invalid in ([], [-1], ["32000"], None):
            with self.assertRaises(ValueError):
                eos_token_ids(invalid, None)


if __name__ == "__main__":
    unittest.main()
