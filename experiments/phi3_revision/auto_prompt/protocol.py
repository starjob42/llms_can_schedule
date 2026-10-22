"""Freeze the separate, validation-guided instruction-refinement protocol.

Building a draft writes nothing. Freezing requires completed engineering
evidence and refuses any existing scientific output. Runtime adapter hashes
are bound later, after the parent study and the full training run complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from core import PROMPTS, SYSTEM_PROMPT
from evaluate import artifact_digest, atomic_json, json_hash, load_dataset, sha256_file

BASE_REVISION = "f3c06aed622e14ca0abf5115094e4fc9a9948f36"
PURPOSE = "SCIENTIFIC_AUTO_PROMPT"
INSTRUCTION_IDS = ("auto_0", "auto_r1_s1", "auto_r1_s2", "auto_r2_s1", "auto_r2_s2")
PROPOSER_CONFIG = {
    "temperature": 0.8, "top_k": 50, "top_p": 0.95,
    "max_new_tokens": 256, "max_attempts_per_instruction": 4,
    "min_chars": 20, "max_chars": 600, "max_instruction_tokens": 128,
    "seed": 42, "precision": "float16", "adapter": None,
}
DECODING = {
    "temperature": 1.0, "top_k": 50, "top_p": 0.95, "seed": 42,
    "validation_candidates": 3, "validation_batch_size": 3,
    "test_candidates": 10, "test_batch_size": 5,
    "precision": "float16", "attention_implementation": "sdpa",
    "max_new_tokens_rule": "min(8192,48*n_operations+128)",
    "generation_seed_rule": "sha256(instance_id:42:batch_index) first 8 bytes mod (2**31-1)",
    "stopping": "all official base generation_config EOS values; fixed candidate count",
}
SOURCE_FILES = (
    PARENT / "core.py", PARENT / "evaluate.py", PARENT / "train.py", PARENT / "run_study.py", PARENT / "report.py",
    HERE / "protocol.py", HERE / "optimizer.py", HERE / "evaluate.py", HERE / "report.py",
)


def protocol_digest(value):
    return json_hash({key: item for key, item in value.items() if key != "protocol_sha256"})


def self_hashed(path, key):
    value = json.loads(Path(path).read_text())
    if value.get(key) != json_hash({name: item for name, item in value.items() if name != key}):
        raise ValueError(f"Invalid {key}: {path}")
    return value


def instruction_spec(instruction_id, text, origin, token_count):
    """Create immutable text metadata; the optimizer validates proposal format."""
    if instruction_id not in INSTRUCTION_IDS or not isinstance(text, str) or not text:
        raise ValueError("Unknown instruction or empty text")
    order = INSTRUCTION_IDS.index(instruction_id)
    expected_round, expected_slot = (0, 0) if order == 0 else ((order - 1) // 2 + 1, (order - 1) % 2 + 1)
    if set(origin) != {"round", "slot", "attempt"} or (origin["round"], origin["slot"]) != (expected_round, expected_slot):
        raise ValueError("Instruction origin differs from its fixed creation order")
    if type(origin["attempt"]) is not int or not (0 <= origin["attempt"] <= 4) or (origin["attempt"] == 0) != (order == 0):
        raise ValueError("Invalid instruction proposal attempt")
    if type(token_count) is not int or not 0 < token_count <= 128:
        raise ValueError("Instruction token count is outside the protocol")
    if order == 0 and text != PROMPTS[0].text:
        raise ValueError("Initial instruction must reproduce seen_1 exactly")
    return {"id": instruction_id, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "origin": dict(origin), "token_count": token_count, "creation_order": order}


def verify_instruction(spec):
    expected = instruction_spec(spec["id"], spec["text"], spec["origin"], spec["token_count"])
    if spec != expected:
        raise ValueError("Instruction metadata or text hash changed")
    return spec


def _verify_parent(study_dir):
    study_dir = Path(study_dir).resolve()
    study = self_hashed(study_dir / "STUDY.json", "protocol_sha256")
    for name, digest in study["artifact_sha256"].items():
        if sha256_file(study_dir / name) != digest:
            raise ValueError(f"Parent study artifact changed: {name}")
    if (study["test_count"], study["prompt_validation_count"], study["checkpoint_steps"]) != (216, 72, [0, 461, 922, 1383, 1844]):
        raise ValueError("Unexpected parent study population or checkpoints")
    snapshot = json.loads((study_dir / "PREPARATION_SPLIT.json").read_text())
    prepared = json.loads(Path(study["prepared_manifest"]).read_text())
    if prepared.get("status") != "prepared" or any(prepared[key] != value for key, value in snapshot.items()):
        raise ValueError("Parent prepared split changed")
    if (prepared["training_count"], prepared["validation_count"]) != (118000, 2000):
        raise ValueError("Parent training split differs from 118k/2k")
    allowed = set(snapshot["validation_indices"])
    populations = {}
    for name, key, indices_key, split, source_key, count in (
        ("test.jsonl", "test_instance_ids", "test_source_indices", "released_test", "test_source_sha256", 216),
        ("prompt_validation.jsonl", "prompt_validation_instance_ids", "prompt_validation_source_indices", "training_source_validation", "training_source_sha256", 72),
    ):
        items, source_count = load_dataset(study_dir / name)
        identities = [item.instance_id for item in items]
        if len(items) != count or source_count != count or identities != study[key]:
            raise ValueError("Parent canonical identities or population changed")
        rows = [json.loads(line) for line in (study_dir / name).read_text().splitlines() if line.strip()]
        if [row.get("source_index") for row in rows] != study[indices_key]:
            raise ValueError("Parent source indices changed")
        if any(row.get("source_split") != split or row.get("source_sha256") != snapshot[source_key] for row in rows):
            raise ValueError("Parent source provenance changed")
        if name == "prompt_validation.jsonl" and any(row["source_index"] not in allowed for row in rows):
            raise ValueError("Prompt selection is not held out from training")
        populations[name] = identities
    if set(populations["test.jsonl"]) & set(populations["prompt_validation.jsonl"]):
        raise ValueError("Validation/test overlap")
    return study, prepared


def _engineering_evidence(paths):
    records = []
    for path in paths:
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        if value.get("status") != "verified" or value.get("purpose") != "ENGINEERING_ONLY":
            raise ValueError("Engineering evidence must be verified and explicitly excluded from scientific results")
        if not isinstance(value.get("checks"), dict) or not value["checks"] or any(check is not True for check in value["checks"].values()):
            raise ValueError("Engineering evidence has missing or failed checks")
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return records


def build_protocol(study_dir, runs_dir, base_path, output_dir, engineering_evidence=()):
    """Return a complete draft without launching jobs or writing a freeze file."""
    study_dir, runs_dir, base_path, output_dir = map(lambda path: Path(path).resolve(),
                                                   (study_dir, runs_dir, base_path, output_dir))
    study, prepared = _verify_parent(study_dir)
    source_files = {str(path): sha256_file(path) for path in SOURCE_FILES}
    prepared_path = Path(study["prepared_manifest"]).resolve()
    data_files = {str(prepared_path): sha256_file(prepared_path)}
    for key in ("training_source", "test_source"):
        path = Path(prepared[key]).resolve()
        if sha256_file(path) != prepared[key + "_sha256"]:
            raise ValueError("Released source data changed")
        data_files[str(path)] = prepared[key + "_sha256"]
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        path = prepared_path.parent / name
        digest = prepared["tokenization"][split]["sha256"]
        if sha256_file(path) != digest:
            raise ValueError("Parent training tokenization changed")
        data_files[str(path)] = digest
    evidence = _engineering_evidence(engineering_evidence)
    value = {
        "schema_version": 1, "study": "validation_guided_automated_instruction_generation_refinement",
        "status": "frozen_before_auto_prompt_outcomes" if evidence else "engineering_draft_excluded",
        "purpose": PURPOSE if evidence else "ENGINEERING_ONLY",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_study": str(study_dir / "STUDY.json"), "parent_protocol_sha256": study["protocol_sha256"],
        "runs_dir": str(runs_dir), "parent_complete": str(runs_dir / "manager/COMPLETE"),
        "training_complete": str(runs_dir / "main/training_complete.json"),
        "base_path": str(base_path), "base_model": "microsoft/Phi-3-mini-128k-instruct", "base_revision": BASE_REVISION,
        "base_artifacts": artifact_digest(base_path),
        "adapter_path": str(runs_dir / "main/checkpoint-1844"), "required_training_step": 1844,
        "validation_file": str(study_dir / "prompt_validation.jsonl"),
        "test_file": str(study_dir / "test.jsonl"),
        "validation_instance_ids": study["prompt_validation_instance_ids"], "test_instance_ids": study["test_instance_ids"],
        "output_dir": str(output_dir), "instruction_ids": list(INSTRUCTION_IDS),
        "initial_instruction": PROMPTS[0].text, "system_prompt": SYSTEM_PROMPT,
        "user_template": "{instruction}\n{problem}", "chat_framing": "native tokenizer add_generation_prompt=True",
        "rounds": 2, "new_instructions_per_round": 2,
        "validation_instances": 72, "validation_candidates_total": 1080,
        "test_instances": 216, "test_candidates_total": 2160,
        "proposer": dict(PROPOSER_CONFIG), "decoding": dict(DECODING),
        "selection_rule": "maximum feasible candidates; minimum conditional mean candidate reference gap; stable creation order",
        "round_rule": "both proposals in a round use the same previous-round incumbent and its aggregate validation feedback; select among all instructions so far",
        "proposal_seed_rule": "sha256(phi3_auto_prompt:42:round:slot:attempt) first 8 bytes mod (2**31-1)",
        "proposal_acceptance": "first valid strict JSON instruction per slot; 20..600 printable ASCII characters, <=128 official tokenizer tokens, single line, no surrounding whitespace or role/special delimiters, normalized unique among accepted instructions; fail closed after four attempts",
        "proposer_feedback_allowlist": ["n_instances", "candidate_count", "n_candidates", "feasible_candidates", "conditional_mean_gap_percent", "error_counts"],
        "test_access": "no test outcomes or records used for proposals, validation selection, or retries; selected instruction frozen before a fresh selected-instruction test evaluation",
        "comparison_rule": "after frozen selection and fresh test completion only, compare with parent seen_1 and fixed-registry selected prompt on identical frozen216/k10 populations",
        "limitations": "one training seed and paired decoding seed scheme; no DSPy; no per-instance instruction adaptation; finite two-round generation/refinement budget",
        "source_sha256": source_files, "data_sha256": data_files,
        "engineering_evidence": evidence,
    }
    value["protocol_sha256"] = protocol_digest(value)
    return value


def _validate_shape(value, allow_engineering_draft=False):
    if value.get("protocol_sha256") != protocol_digest(value):
        raise ValueError("Automatic-prompt protocol hash is invalid")
    draft = allow_engineering_draft and value.get("status") == "engineering_draft_excluded" and value.get("purpose") == "ENGINEERING_ONLY"
    expected = {"schema_version": 1, "purpose": "ENGINEERING_ONLY" if draft else PURPOSE,
                "status": "engineering_draft_excluded" if draft else "frozen_before_auto_prompt_outcomes",
                "base_revision": BASE_REVISION, "instruction_ids": list(INSTRUCTION_IDS),
                "rounds": 2, "new_instructions_per_round": 2, "required_training_step": 1844,
                "validation_instances": 72, "validation_candidates_total": 1080,
                "test_instances": 216, "test_candidates_total": 2160,
                "initial_instruction": PROMPTS[0].text, "system_prompt": SYSTEM_PROMPT,
                "proposer": PROPOSER_CONFIG, "decoding": DECODING}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Automatic-prompt protocol differs from the approved fixed design")
    if len(set(value["validation_instance_ids"])) != 72 or len(set(value["test_instance_ids"])) != 216:
        raise ValueError("Unexpected automatic-prompt population")
    if set(value["validation_instance_ids"]) & set(value["test_instance_ids"]):
        raise ValueError("Automatic-prompt validation/test overlap")


def load_protocol(path, verify_files=True):
    path = Path(path)
    if path.is_dir():
        path = path / "PROTOCOL.json"
    value = json.loads(path.read_text())
    _validate_shape(value, allow_engineering_draft=not verify_files)
    if verify_files:
        study_dir = Path(value["parent_study"]).parent.resolve()
        study, prepared = _verify_parent(study_dir)
        if study["protocol_sha256"] != value["parent_protocol_sha256"]:
            raise ValueError("Automatic-prompt protocol parent changed")
        for name, key in (("prompt_validation_instance_ids", "validation_instance_ids"), ("test_instance_ids", "test_instance_ids")):
            if study[name] != value[key]:
                raise ValueError("Automatic-prompt protocol populations differ from its parent")
        runs_dir = Path(value["runs_dir"]).resolve()
        expected_locations = {"parent_study": study_dir / "STUDY.json",
                              "validation_file": study_dir / "prompt_validation.jsonl", "test_file": study_dir / "test.jsonl",
                              "adapter_path": runs_dir / "main/checkpoint-1844",
                              "parent_complete": runs_dir / "manager/COMPLETE", "training_complete": runs_dir / "main/training_complete.json"}
        if any(Path(value[key]).resolve() != location for key, location in expected_locations.items()):
            raise ValueError("Automatic-prompt paths differ from the frozen parent layout")
        expected_paths = {str(path.resolve()) for path in SOURCE_FILES}
        if set(value["source_sha256"]) != expected_paths:
            raise ValueError("Automatic-prompt frozen source file set changed")
        prepared_path = Path(study["prepared_manifest"]).resolve()
        expected_data = {str(prepared_path): sha256_file(prepared_path)}
        for key in ("training_source", "test_source"):
            expected_data[str(Path(prepared[key]).resolve())] = prepared[key + "_sha256"]
        for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
            expected_data[str(prepared_path.parent / name)] = prepared["tokenization"][split]["sha256"]
        if value["data_sha256"] != expected_data:
            raise ValueError("Automatic-prompt source/tokenization hash set changed")
        for path, digest in {**value["source_sha256"], **value["data_sha256"]}.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"Automatic-prompt frozen source or data changed: {path}")
        if artifact_digest(Path(value["base_path"])) != value["base_artifacts"]:
            raise ValueError("Pinned unadapted proposer/evaluator base artifacts changed")
        if not value["engineering_evidence"] or _engineering_evidence(item["path"] for item in value["engineering_evidence"]) != value["engineering_evidence"]:
            raise ValueError("Automatic-prompt engineering evidence is absent or changed")
    return value


def freeze_protocol(protocol, path):
    """Publish an immutable protocol after engineering tests and before outcomes."""
    if not protocol["engineering_evidence"]:
        raise ValueError("Complete engineering tests/pilots before protocol freeze")
    _validate_shape(protocol)
    if _engineering_evidence(item["path"] for item in protocol["engineering_evidence"]) != protocol["engineering_evidence"]:
        raise ValueError("Engineering evidence changed before freeze")
    output = Path(protocol["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Scientific outputs already exist; a protocol cannot be frozen retrospectively")
    path = Path(path)
    if path.exists():
        existing = load_protocol(path)
        if existing != protocol:
            raise ValueError("Refusing to overwrite a different frozen automatic-prompt protocol")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, protocol)
    try:
        return load_protocol(path)
    except Exception:
        path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--base-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engineering-evidence", action="append", default=[], type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    value = build_protocol(args.study_dir, args.runs_dir, args.base_path, args.output_dir, args.engineering_evidence)
    if args.freeze:
        freeze_protocol(value, args.freeze)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
