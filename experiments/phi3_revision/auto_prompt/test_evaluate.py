"""Excluded CPU fake-engine checks for dynamic registry isolation and guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.phi3_revision import core, evaluate as primary
from experiments.phi3_revision.auto_prompt import evaluate as dynamic


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
    def __init__(self):
        self.calls = []
    def encode(self, text):
        self.user_text = text
        return FakeEncoded()
    def generate(self, encoded, count, seed, max_new_tokens):
        self.calls.append((count, seed, max_new_tokens))
        return [{"output": OUTPUT, "input_tokens": 100, "output_tokens": 80,
                 "batch_seconds": 1.0, "amortized_seconds": 1.0 / count,
                 "hit_max_new_tokens": False} for _ in range(count)]


class DynamicEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base, self.adapter, self.study = [self.root / name for name in ("base", "adapter", "study")]
        for path in (self.base, self.adapter, self.study):
            path.mkdir()
        (self.base / "model.safetensors").write_bytes(b"excluded fake weights")
        (self.base / "config.json").write_text('{"model_type":"phi3"}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"excluded fake adapter")
        (self.adapter / "adapter_config.json").write_text("{}")
        parent = {"artifact_sha256": {}, "test_instance_ids": ["heldout_test"], "prompt_validation_instance_ids": ["heldout_validation"]}
        parent["protocol_sha256"] = dynamic._hash(parent)
        (self.study / "STUDY.json").write_text(json.dumps(parent))
        self.protocol = {"parent_study": str(self.study / "STUDY.json"), "parent_protocol_sha256": parent["protocol_sha256"],
                         "base_path": str(self.base), "adapter_path": str(self.adapter),
                         "output_dir": str(self.root / "production"), "validation_file": str(self.study / "prompt_validation.jsonl"),
                         "test_file": str(self.study / "test.jsonl"),
                         "validation_instance_ids": ["heldout_validation"], "test_instance_ids": ["heldout_test"]}
        self.protocol["protocol_sha256"] = dynamic._hash(self.protocol)
        self.data = self.root / "excluded_toy.jsonl"
        self.data.write_text(json.dumps({"matrix": MATRIX, "prompt_machines_first": TEXT, "output": OUTPUT}) + "\n")
        self.spec = {"id": "auto_pilot", "text": core.PROMPTS[0].text,
                     "text_sha256": hashlib.sha256(core.PROMPTS[0].text.encode()).hexdigest(),
                     "origin": {"round": 0, "slot": 0, "attempt": 0}, "token_count": 24, "creation_order": 0}
        self.out = self.root / "excluded_evaluation"
        self.arguments = {"protocol_path": self.root / "DRAFT_PROTOCOL.json", "split": "validation", "study_dir": self.study, "pilot": True}
        self.protocol_patch = patch.object(dynamic, "_read_protocol", return_value=self.protocol)
        self.protocol_patch.start()

    def tearDown(self):
        self.protocol_patch.stop()
        self.tmp.cleanup()

    def run_fake(self, engine=None):
        engine = engine or FakeEngine()
        dynamic._run_in_process(self.spec, self.data, self.adapter, self.out, self.base, engine=engine, **self.arguments)
        return engine

    def verify(self):
        return dynamic.verify_instruction(self.spec, self.data, self.adapter, self.out, self.base, **self.arguments)

    def test_private_registry_changes_every_native_prompt_path_without_mutating_shared_modules(self):
        shared_core, shared_primary = core.PROMPTS, primary.PROMPTS
        with dynamic._private_evaluator(self.spec) as module:
            config = module.EvaluationConfig("b", "d", "o", prompt_ids=(self.spec["id"],))
            config.validate()
            self.assertEqual(module.evaluation_user_prompt(self.spec["id"], "INSTANCE"), self.spec["text"] + "\nINSTANCE")
            self.assertEqual(len(module.PROMPTS), 1)
            self.assertIs(core.PROMPTS, shared_core)
            self.assertIs(primary.PROMPTS, shared_primary)
            with self.assertRaises(ValueError):
                primary.EvaluationConfig("b", "d", "o", prompt_ids=(self.spec["id"],)).validate()
        self.assertIs(core.PROMPTS, shared_core)
        self.assertIs(primary.PROMPTS, shared_primary)

    def test_complete_fake_engine_pilot_preserves_native_slots_and_explicit_text_hash(self):
        engine = self.run_fake()
        verified = self.verify()
        self.assertEqual(engine.calls[0][0], 3)
        self.assertEqual(engine.calls[0][2], 320)
        self.assertEqual(verified["feedback"]["n_candidates"], 3)
        self.assertEqual(verified["feedback"]["feasible_candidates"], 3)
        self.assertTrue(verified["feedback"]["pilot_excluded"])
        self.assertEqual(verified["manifest"]["run_kind"], "PILOT_LIMITED")
        self.assertEqual(verified["manifest"]["frozen_prompts"][0]["text"], self.spec["text"])
        self.assertTrue(verified["manifest"]["frozen_prompts"][0]["seen_in_training"])
        self.assertTrue(verified["instruction_manifest"]["seen_in_parent_training"])
        self.assertEqual(verified["instruction_manifest"]["exact_parent_prompt_matches"], ["seen_1"])
        self.assertEqual(verified["instruction_manifest"]["instruction_text_sha256"], self.spec["text_sha256"])
        self.assertEqual(set(verified["receipt"]), {"evaluation_manifest_sha256", "candidates_sha256", "complete_sha256", "instruction_manifest_sha256"})

    def test_complete_resume_avoids_generation_and_revalidates_cached_claims(self):
        self.run_fake()
        path = self.out / "candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["validation"]["feasible"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        engine = self.run_fake()
        self.assertEqual(engine.calls, [])
        self.assertEqual(self.verify()["feedback"]["feasible_candidates"], 3)

    def test_interrupted_batch_replays_original_full_batch_and_seed(self):
        first_engine = self.run_fake()
        path = self.out / "candidates.jsonl"
        first = path.read_text().splitlines()[0]
        path.write_text(first + "\n")
        engine = self.run_fake()
        self.assertEqual(engine.calls, first_engine.calls)
        self.assertEqual(len(self.verify()["records"]), 3)

    def test_tampered_instruction_and_raw_budget_are_rejected(self):
        self.run_fake()
        bad = deepcopy(self.spec)
        bad["text"] += " Use a short answer."
        bad["text_sha256"] = hashlib.sha256(bad["text"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            dynamic._run_in_process(bad, self.data, self.adapter, self.out, self.base, engine=FakeEngine(), **self.arguments)
        path = self.out / "candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["max_new_tokens"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "fixed generation budget"):
            self.verify()

    def test_missing_completion_does_not_trigger_generation_during_verification(self):
        self.run_fake()
        (self.out / "COMPLETE").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_test_selection_must_exist_and_match_before_generation(self):
        with self.assertRaises(FileNotFoundError):
            dynamic._selection(self.protocol, self.spec)
        selected = {"status": "frozen_before_auto_test", "protocol_sha256": self.protocol["protocol_sha256"],
                    "selected_instruction": self.spec, "selected_instruction_id": self.spec["id"], "text_sha256": self.spec["text_sha256"]}
        selected["selection_sha256"] = dynamic._hash(selected)
        target = Path(self.protocol["output_dir"])
        target.mkdir()
        (target / "SELECTED.json").write_text(json.dumps(selected))
        self.assertEqual(dynamic._selection(self.protocol, self.spec), selected)
        wrong = dict(self.spec, id="auto_other")
        with self.assertRaisesRegex(ValueError, "frozen validation-only selection"):
            dynamic._selection(self.protocol, wrong)

    def test_scientific_fake_engine_and_toy_overlap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "only for excluded"):
            dynamic._run_in_process(self.spec, self.data, self.adapter, self.out, self.base,
                                    engine=FakeEngine(), **dict(self.arguments, pilot=False))
        self.protocol["test_instance_ids"] = [core.canonical_hash(core.parse_matrix(MATRIX))]
        with self.assertRaisesRegex(ValueError, "excluded from both"):
            self.run_fake()

    def test_bad_hash_role_tokens_and_production_origins_are_rejected(self):
        for bad in (dict(self.spec, text_sha256="wrong"), dict(self.spec, text="<|system|>" + self.spec["text"])):
            with self.assertRaises(ValueError):
                dynamic._validate_spec(bad, pilot=True)
        initial = dict(self.spec, id="auto_0")
        dynamic._validate_spec(initial)
        with self.assertRaises(ValueError):
            dynamic._validate_spec(dict(initial, creation_order=1))


if __name__ == "__main__":
    unittest.main()
