"""CPU-only protocol/schema tests; importing train never loads torch or a model."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from train import (PreflightError, RunConfig, TokenizedJSONL, assert_resume_compatible,
                   base_model_fingerprint, check_output_directory, examples_at_step, load_safe_training_state,
                   parse_milestones, read_record, resolve_targets, save_safe_training_state)


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = RunConfig(str(self.root), str(self.root / "train.jsonl"), str(self.root / "out"))

    def tearDown(self):
        self.temp.cleanup()

    def test_default_controlled_schedule(self):
        self.config.validate()
        self.assertEqual(self.config.effective_batch_size, 64)
        self.assertEqual(self.config.milestones, (0, 461, 922, 1383, 1844))
        self.assertEqual((self.config.lora_rank, self.config.lora_alpha), (128, 256))
        self.assertNotEqual(self.config.protocol, replace(self.config, precision="bf16-lora").protocol)
        self.assertTrue(replace(self.config, pilot=True).protocol.startswith("pilot_only:"))

    def test_invalid_milestones_rejected(self):
        for value in ("461,1844", "0,461,461,1844", "0,1844,922", "0,461", "0,-1,1844", "0,no,1844"):
            with self.subTest(value=value), self.assertRaises(PreflightError):
                parse_milestones(value, 1844)

    def test_batch_and_precision_cannot_silently_change(self):
        for config in (replace(self.config, gradient_accumulation_steps=1),
                       replace(self.config, precision="fp32"),
                       replace(self.config, learning_rate=float("nan")),
                       replace(self.config, lora_dropout=1)):
            with self.subTest(config=config), self.assertRaises(PreflightError):
                config.validate()

    def test_strict_unpadded_schema(self):
        valid = read_record(b'{"input_ids":[1,2,3],"instance_id":"ignored"}', "test:1", 3)
        self.assertEqual(valid["attention_mask"], [1, 1, 1])
        for row in ({"input_ids": [1, 2, 3, 4]}, {"input_ids": []}, {"input_ids": [True, 2]},
                    {"input_ids": [-1, 2]}, {"input_ids": [1, 2], "attention_mask": [1, 0]},
                    {"input_ids": [1, 2], "attention_mask": [1]},
                    {"input_ids": [1, 2], "labels": [1, 2]}, {"input_ids": [1]}):
            with self.subTest(row=row), self.assertRaises(PreflightError):
                read_record(json.dumps(row).encode(), "test:2", 3)

    def test_jsonl_random_access_and_manifest_hash(self):
        path = self.root / "train.jsonl"
        path.write_text('{"input_ids":[1,2]}\n{"input_ids":[3,4,5]}\n')
        data = TokenizedJSONL(path, 3)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[1]["input_ids"], [3, 4, 5])
        self.assertEqual(data[0]["input_ids"], [1, 2])
        self.assertEqual(data.summary()["tokens"], 5)
        self.assertEqual(data.max_token_id, 5)
        self.assertEqual(len(data.sha256), 64)
        data._handle.close()
        path.write_text('{"input_ids":[1,2,3,4]}\n')
        with self.assertRaisesRegex(PreflightError, "no truncation"):
            TokenizedJSONL(path, 3)

    def test_phi_targets_use_actual_intersection(self):
        names = ["model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
                 "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"]
        actual = resolve_targets(names)
        self.assertEqual(actual["selected"], ["qkv_proj", "o_proj"])
        self.assertEqual(actual["absent"], ["fc1", "fc2"])
        with self.assertRaises(PreflightError):
            resolve_targets(["model.q_proj", "model.o_proj"])
        with self.assertRaises(PreflightError):
            resolve_targets(names + ["model.fc1"])

    def test_no_overwrite_and_resume_requires_complete_state(self):
        out = self.root / "out"
        self.assertIsNone(check_output_directory(out, None))
        out.mkdir()
        (out / "run_manifest.json").write_text("{}")
        with self.assertRaises(PreflightError):
            check_output_directory(out, None)
        with self.assertRaises(PreflightError):
            check_output_directory(out, "latest")
        full = out / "checkpoint-461"
        full.mkdir()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                     "training_state.json", "training_state.safetensors", "experiment_checkpoint.json"):
            (full / name).touch()
        incomplete = out / "checkpoint-922"
        incomplete.mkdir()
        self.assertEqual(check_output_directory(out, "latest"), full)
        with self.assertRaises(PreflightError):
            check_output_directory(out, str(incomplete))

    def test_resume_refuses_schedule_or_data_change(self):
        manifest = {"config": {"total_steps": 1844}, "train_data": {"sha256": "abc"}, "validation_data": None}
        assert_resume_compatible(manifest, manifest.copy())
        for changed in ({**manifest, "config": {"total_steps": 922}},
                        {**manifest, "train_data": {"sha256": "xyz"}}):
            with self.assertRaises(PreflightError):
                assert_resume_compatible(manifest, changed)

    def test_actual_examples_include_partial_final_batch(self):
        self.assertEqual(examples_at_step(0, 118000), 0)
        self.assertEqual(examples_at_step(461, 118000), 29504)
        self.assertEqual(examples_at_step(1844, 118000), 118000)
        self.assertEqual(examples_at_step(1845, 118000), 118064)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "CPU torch/safetensors unavailable in this Python")
    def test_safe_state_roundtrip_without_pickle(self):
        import numpy as np
        import random
        import torch
        state = {"optimizer": {"state": {0: {"step": 3, "exp_avg": torch.tensor([1.0, 2.0])}},
                               "param_groups": [{"params": [0], "lr": 0.0002}]},
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.random.get_rng_state(), "scaler": None}
        save_safe_training_state(self.root, state)
        restored = load_safe_training_state(self.root)
        self.assertTrue(torch.equal(restored["optimizer"]["state"][0]["exp_avg"], torch.tensor([1.0, 2.0])))
        self.assertEqual(restored["python_rng"], state["python_rng"])
        self.assertTrue(np.array_equal(restored["numpy_rng"][1], state["numpy_rng"][1]))
        self.assertTrue(torch.equal(restored["torch_rng"], state["torch_rng"]))
        self.assertEqual(restored["optimizer"]["param_groups"], state["optimizer"]["param_groups"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "CPU torch/safetensors unavailable in this Python")
    def test_base_fingerprint_checks_and_hashes_all_weight_bytes(self):
        import hashlib
        import torch
        from safetensors.torch import save_file
        for name, value in (("config.json", {"model_type": "phi3"}),
                            ("tokenizer_config.json", {}), ("tokenizer.json", {})):
            (self.root / name).write_text(json.dumps(value))
        with self.assertRaisesRegex(PreflightError, "missing weight shard"):
            base_model_fingerprint(str(self.root))
        target = self.root / "model.safetensors"
        save_file({"weight": torch.tensor([1.0, 2.0])}, str(target))
        result = base_model_fingerprint(str(self.root))
        self.assertEqual(result["weight_file_sha256"][target.name], hashlib.sha256(target.read_bytes()).hexdigest())
        target.write_bytes(target.read_bytes()[:-1])
        with self.assertRaisesRegex(PreflightError, "incomplete or invalid"):
            base_model_fingerprint(str(self.root))


if __name__ == "__main__":
    unittest.main()
