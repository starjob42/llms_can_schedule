"""Focused CPU checks of the separate protocol's freeze and provenance guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import protocol


class InstructionMetadataTests(unittest.TestCase):
    def test_creation_order_text_and_attempts_are_bound(self):
        initial=protocol.instruction_spec("auto_0",protocol.PROMPTS[0].text,{"round":0,"slot":0,"attempt":0},25)
        self.assertEqual(protocol.verify_instruction(initial),initial)
        generated=protocol.instruction_spec("auto_r2_s1","Schedule all operations and report the makespan.",{"round":2,"slot":1,"attempt":3},10)
        self.assertEqual(generated["creation_order"],3)
        for changed in [dict(initial,text="a changed initial instruction"),dict(generated,creation_order=1),dict(generated,text_sha256="forged")]:
            with self.assertRaises(ValueError):protocol.verify_instruction(changed)
        with self.assertRaises(ValueError):
            protocol.instruction_spec("auto_r1_s2",generated["text"],{"round":1,"slot":1,"attempt":1},10)
        with self.assertRaises(ValueError):
            protocol.instruction_spec("auto_r1_s1",generated["text"],{"round":1,"slot":1,"attempt":0},10)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="auto_protocol_unit_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.study_dir=self.root/"study";self.study_dir.mkdir()
        self.prepared_path=self.root/"manifest.json"
        train_source=self.root/"source_train.json";train_source.write_text("fixture training source\n")
        test_source=self.root/"source_test.json";test_source.write_text("fixture test source\n")
        train=self.root/"train.jsonl";train.write_text('{"input_ids":[1,2]}\n')
        validation=self.root/"validation.jsonl";validation.write_text('{"input_ids":[3,4]}\n')
        self.prepared={"training_source":str(train_source),"training_source_sha256":protocol.sha256_file(train_source),
                       "test_source":str(test_source),"test_source_sha256":protocol.sha256_file(test_source),
                       "tokenization":{"train":{"sha256":protocol.sha256_file(train)},"validation":{"sha256":protocol.sha256_file(validation)}}}
        self.prepared_path.write_text(json.dumps(self.prepared))
        self.study={"prepared_manifest":str(self.prepared_path),"protocol_sha256":"parent fixture digest",
                    "prompt_validation_instance_ids":[f"val_{i}" for i in range(72)],
                    "test_instance_ids":[f"test_{i}" for i in range(216)]}
        self.code=self.root/"frozen_code.py";self.code.write_text("# fixed fixture source\n")
        self.base=self.root/"base";self.base.mkdir()
        (self.base/"config.json").write_text('{}\n')
        (self.base/"model.safetensors").write_bytes(b"not real model weights; file hashing only")
        self.proof=self.root/"engineering.json"
        self.proof.write_text(json.dumps({"status":"verified","purpose":"ENGINEERING_ONLY","checks":{"cpu":True,"gpu_pilot":True}}))
        self.output=self.root/"production"
        self.path=self.root/"PROTOCOL.json"
        self.patch_parent=patch.object(protocol,"_verify_parent",return_value=(self.study,self.prepared))
        self.patch_files=patch.object(protocol,"SOURCE_FILES",(self.code,))
        self.patch_parent.start();self.addCleanup(self.patch_parent.stop)
        self.patch_files.start();self.addCleanup(self.patch_files.stop)

    def build(self,proof=True):
        return protocol.build_protocol(self.study_dir,self.root/"runs",self.base,self.output,[self.proof] if proof else [])

    def test_freeze_binds_source_data_models_and_engineering_evidence(self):
        value=self.build()
        self.assertFalse(self.path.exists())
        frozen=protocol.freeze_protocol(value,self.path)
        self.assertEqual(frozen["validation_candidates_total"],1080)
        self.assertEqual(frozen["test_candidates_total"],2160)
        self.assertIsNone(frozen["proposer"]["adapter"])
        self.code.write_text("# changed after freeze\n")
        with self.assertRaisesRegex(ValueError,"source or data changed"):
            protocol.load_protocol(self.path)

    def test_missing_failed_or_scientific_engineering_proof_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"engineering"):
            protocol.freeze_protocol(self.build(proof=False),self.path)
        self.proof.write_text(json.dumps({"status":"verified","purpose":"SCIENTIFIC_AUTO_PROMPT","checks":{"cpu":True}}))
        with self.assertRaisesRegex(ValueError,"excluded"):
            self.build()
        self.proof.write_text(json.dumps({"status":"verified","purpose":"ENGINEERING_ONLY","checks":{"gpu_pilot":False}}))
        with self.assertRaisesRegex(ValueError,"failed"):
            self.build()
        self.assertFalse(self.path.exists())

    def test_retrospective_freeze_and_design_drift_are_rejected(self):
        value=self.build()
        self.output.mkdir();(self.output/"RUN.json").write_text("existing outcomes\n")
        with self.assertRaisesRegex(ValueError,"retrospectively"):
            protocol.freeze_protocol(value,self.path)
        changed=copy.deepcopy(value);changed["proposer"]["max_attempts_per_instruction"]=99
        changed["protocol_sha256"]=protocol.protocol_digest(changed)
        with self.assertRaisesRegex(ValueError,"approved fixed design"):
            protocol.freeze_protocol(changed,self.path)
        self.assertFalse(self.path.exists())

    def test_data_hash_set_cannot_drop_the_training_source(self):
        value=self.build()
        del value["data_sha256"][self.prepared["training_source"]]
        value["protocol_sha256"]=protocol.protocol_digest(value)
        with self.assertRaisesRegex(ValueError,"hash set changed"):
            protocol.freeze_protocol(value,self.path)
        self.assertFalse(self.path.exists())

    def test_model_and_engineering_mutation_after_freeze_are_rejected(self):
        protocol.freeze_protocol(self.build(),self.path)
        (self.base/"model.safetensors").write_bytes(b"changed fixture weights")
        with self.assertRaisesRegex(ValueError,"base artifacts changed"):
            protocol.load_protocol(self.path)


if __name__=="__main__":unittest.main()
