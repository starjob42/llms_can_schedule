#!/usr/bin/env python3
"""Freeze outcome-blind, stratified evaluation and prompt-validation sets.

This consumes the audited preparation manifest before or after tokenization.
Selection uses only dimensions and a salted canonical instance hash. Solver
labels, generated schedules, and model scores never determine membership.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .core import PROMPTS, SYSTEM_PROMPT, canonical_hash, parse_instance, parse_matrix
except ImportError:
    from core import PROMPTS, SYSTEM_PROMPT, canonical_hash, parse_instance, parse_matrix


SEED = 42
CHECKPOINT_STEPS = [0, 461, 922, 1383, 1844]
TEST_PER_SHAPE = 6
VALIDATION_PER_SHAPE = 2


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(record: Mapping) -> tuple[str, tuple[int, int]]:
    """Read only problem inputs; discard the matrix's solver-label trailer."""
    if record.get("matrix"):
        lines = [line.strip() for line in record["matrix"].splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty problem matrix")
        n_jobs = int(lines[0].split()[0])
        instance = parse_matrix("\n".join(lines[:n_jobs + 1]))
    else:
        instance = parse_instance(record["prompt_jobs_first"])
    identity = canonical_hash(instance)
    if record.get("instance_id") and record["instance_id"] != identity:
        raise ValueError("Stored instance_id disagrees with canonical problem inputs")
    return identity, (instance.n_jobs, instance.n_machines)


@dataclass(frozen=True)
class Candidate:
    instance_id: str
    shape: tuple[int, int]
    source_index: int
    record: Mapping


def candidate(record: Mapping, source_index: int) -> Candidate:
    identity, shape = input_identity(record)
    return Candidate(identity, shape, source_index, record)


def stratified_selection(
    records: Sequence[Candidate], required_shapes: Iterable[tuple[int, int]],
    per_shape: int, *, seed: int = SEED, salt: str,
) -> list[Candidate]:
    """Select lowest salted SHA256 ranks independently in every required shape."""
    if per_shape < 1:
        raise ValueError("per_shape must be positive")
    shapes = set(required_shapes)
    if not shapes:
        raise ValueError("At least one required shape is necessary")
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        if row.instance_id in seen:
            raise ValueError("Duplicate canonical source instance")
        seen.add(row.instance_id)
        grouped[row.shape].append(row)
    selected = []
    for shape in sorted(shapes):
        pool = grouped[shape]
        if len(pool) < per_shape:
            raise ValueError(f"Shape {shape} has {len(pool)} eligible records; need {per_shape}")
        def rank(row):
            material = f"phi3_revision:{seed}:{salt}:{row.instance_id}".encode()
            return hashlib.sha256(material).hexdigest(), row.instance_id
        selected.extend(sorted(pool, key=rank)[:per_shape])
    return selected


def validate_partition(training_indices, validation_indices, source_count: int) -> None:
    training = list(training_indices)
    validation = list(validation_indices)
    for indices in (training, validation):
        if any(type(index) is not int or not 0 <= index < source_count for index in indices):
            raise ValueError("Split contains an invalid source index")
        if len(set(indices)) != len(indices):
            raise ValueError("Split contains duplicate source indices")
    if set(training) & set(validation):
        raise ValueError("Training and validation source indices overlap")
    if set(training) | set(validation) != set(range(source_count)):
        raise ValueError("Training/validation indices do not partition the complete source")


def guard_overlap(
    test_rows: Sequence[Candidate], validation_rows: Sequence[Candidate],
    training_instance_ids: Iterable[str], validation_indices: Iterable[int],
) -> None:
    test_ids = {row.instance_id for row in test_rows}
    validation_ids = {row.instance_id for row in validation_rows}
    training_ids = set(training_instance_ids)
    if len(test_ids) != len(test_rows) or len(validation_ids) != len(validation_rows):
        raise ValueError("Frozen split contains duplicate canonical instances")
    if test_ids & validation_ids:
        raise ValueError("Test and prompt-validation canonical instances overlap")
    if (test_ids | validation_ids) & training_ids:
        raise ValueError("Evaluation/prompt-validation canonical instances overlap training")
    allowed = set(validation_indices)
    if any(row.source_index not in allowed for row in validation_rows):
        raise ValueError("Prompt-validation selection contains a training source index")


def freeze_study(prepared_manifest: Path, prepared_test: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite frozen study directory: {output_dir}")
    manifest = json.loads(prepared_manifest.read_text())
    if manifest.get("status") not in {"data_audit_complete", "prepared"}:
        raise ValueError("Preparation data audit must be complete before freezing")
    if manifest.get("seed") != SEED:
        raise ValueError("The frozen study requires preparation seed 42")
    if (manifest.get("training_count"), manifest.get("validation_count"), manifest.get("test_count")) != (118000, 2000, 1800):
        raise ValueError("The full manuscript-recipe study requires 118000 train, 2000 validation, 1800 released test records")
    training_source = Path(manifest["training_source"])
    test_source = Path(manifest["test_source"])
    for path, expected in ((training_source, manifest["training_source_sha256"]),
                           (test_source, manifest["test_source_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"Source SHA256 no longer matches preparation: {path}")

    raw_test = json.loads(test_source.read_text())
    test_rows = [candidate(json.loads(line), index) for index, line in
                 enumerate(prepared_test.read_text().splitlines()) if line.strip()]
    raw_test_ids = [input_identity(row)[0] for row in raw_test]
    if ([row.instance_id for row in test_rows] != raw_test_ids or
            raw_test_ids != manifest["test_instance_ids"]):
        raise ValueError("Prepared test identities/order disagree with the released source or preparation manifest")
    if len(set(raw_test_ids)) != len(raw_test_ids):
        raise ValueError("Released test source contains duplicate canonical instances")
    # Prevent prepared records from silently changing problem inputs or labels.
    for row, raw in zip(test_rows, raw_test):
        if any(row.record.get(key) != value for key, value in raw.items()):
            raise ValueError("Prepared test record differs from its released source")
    shapes = {tuple(map(int, key.split("x"))) for key in manifest["test_dimensions"]}
    if len(shapes) != 36 or Counter(row.shape for row in test_rows) != Counter({shape: 50 for shape in shapes}):
        raise ValueError("Require all 36 released test shapes with exactly 50 records each")
    frozen_test = stratified_selection(test_rows, shapes, TEST_PER_SHAPE, salt="test")
    del raw_test

    print("Auditing training/validation identities before the study is frozen", flush=True)
    source = json.loads(training_source.read_text())
    validate_partition(manifest["training_indices"], manifest["validation_indices"], len(source))
    validation_index_set = set(manifest["validation_indices"])
    source_ids = set()
    training_ids = set()
    validation_pool = []
    for index, record in enumerate(source):
        row = candidate(record, index)
        if row.instance_id in source_ids:
            raise ValueError("Released training source contains duplicate canonical instances")
        source_ids.add(row.instance_id)
        if index in validation_index_set:
            validation_pool.append(row)
        else:
            training_ids.add(row.instance_id)
        if index % 20000 == 0:
            print(f"Audited {index}/{len(source)} original training-source identities", flush=True)
    if source_ids & set(raw_test_ids):
        raise ValueError("Released training source and released test source share canonical instances")
    frozen_validation = stratified_selection(validation_pool, shapes, VALIDATION_PER_SHAPE,
                                             salt="prompt_validation")
    guard_overlap(frozen_test, frozen_validation, training_ids, manifest["validation_indices"])
    if len(frozen_test) != 216 or len(frozen_validation) != 72 or len(training_ids) != 118000:
        raise ValueError("Frozen sample or full-training counts differ from protocol")

    output_dir.mkdir(parents=True)
    artifact_hashes = {}
    for name, rows, split, source_sha in (
        ("test.jsonl", frozen_test, "released_test", manifest["test_source_sha256"]),
        ("prompt_validation.jsonl", frozen_validation, "training_source_validation", manifest["training_source_sha256"]),
    ):
        path = output_dir / name
        with path.open("x") as handle:
            for row in rows:
                output = dict(row.record, instance_id=row.instance_id,
                              source_index=row.source_index, source_split=split,
                              source_sha256=source_sha)
                handle.write(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")
        artifact_hashes[name] = sha256_file(path)
    registry = {"prompts": [asdict(prompt) for prompt in PROMPTS], "system_prompt": SYSTEM_PROMPT}
    if len(registry["prompts"]) != 7:
        raise ValueError("The frozen prompt registry must contain exactly seven prompts")
    registry_path = output_dir / "PROMPTS.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    artifact_hashes[registry_path.name] = sha256_file(registry_path)
    split_keys = ["training_source", "training_source_sha256", "test_source", "test_source_sha256",
                  "training_indices", "validation_indices", "training_count", "validation_count",
                  "test_count", "seed", "split_algorithm"]
    preparation_split = {key: manifest[key] for key in split_keys}
    split_path = output_dir / "PREPARATION_SPLIT.json"
    split_path.write_text(json.dumps(preparation_split, indent=2) + "\n")
    artifact_hashes[split_path.name] = sha256_file(split_path)

    result = {
        "schema_version": 1, "status": "frozen_before_model_outcomes",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "fresh_controlled_phi3_manuscript_recipe",
        "prepared_manifest": str(prepared_manifest.resolve()),
        "prepared_manifest_sha256_at_freeze": sha256_file(prepared_manifest),
        "preparation_split_sha256": json_hash(preparation_split),
        "preparation_note": "Tokenization statistics may be appended later; PREPARATION_SPLIT.json fixes the source hashes and complete split indices.",
        "source_training_count": 120000, "training_count": 118000, "validation_count": 2000,
        "released_test_count": 1800, "test_count": 216, "prompt_validation_count": 72,
        "test_selection_per_shape": 6, "prompt_validation_selection_per_shape": 2,
        "shapes": [f"{a}x{b}" for a, b in sorted(shapes)],
        "selection_seed": SEED,
        "selection_rule": "Within each required shape, take the lowest SHA256('phi3_revision:42:SALT:canonical_instance_id') ranks, breaking ties by canonical_instance_id.",
        "selection_salts": {"test": "test", "prompt_validation": "prompt_validation"},
        "selection_uses_outcomes_or_labels": False,
        "canonical_identity": "SHA256 of dimensions and ordered job/operation/machine/duration tuples; matrix reference trailers and output labels excluded",
        "train_validation_test_canonical_overlap": 0,
        "test_source_indices": [row.source_index for row in frozen_test],
        "test_instance_ids": [row.instance_id for row in frozen_test],
        "prompt_validation_source_indices": [row.source_index for row in frozen_validation],
        "prompt_validation_instance_ids": [row.instance_id for row in frozen_validation],
        "checkpoint_steps": CHECKPOINT_STEPS,
        "training_recipe": {
            "base": "microsoft/Phi-3-mini-128k-instruct", "examples": 118000,
            "epochs": 1, "optimizer_steps": 1844, "seed": 42,
            "micro_batch_size": 2, "gradient_accumulation_steps": 32, "effective_batch_size": 64,
            "lora_rank": 128, "lora_alpha": 256, "lora_dropout": 0.05,
            "lora_targets": ["qkv_proj", "o_proj", "fc1", "fc2"],
            "quantization": "4-bit NF4", "compute_dtype": "float16",
            "learning_rate": 0.0002, "optimizer": "paged_adamw_8bit", "warmup_steps": 1,
            "sequence_cap": 40000, "truncate_sequences": False,
            "reduced_data_training_substitution_allowed": False,
        },
        "decoding": {
            "candidate_count_test": 10, "candidate_count_prompt_validation": 3,
            "batch_size_test": 5, "batch_size_prompt_validation": 3,
            "do_sample": True, "temperature": 1.0, "top_k": 50, "top_p": 0.95,
            "seed": 42, "precision": "float16", "attention_implementation": "sdpa",
            "candidate_stopping": "fixed count; no stopping after a feasible candidate",
            "max_new_tokens_rule": "min(8192,48*n_operations+128)",
            "paired_generation_seed_rule": "SHA256(instance_id:42:batch_index), first 8 bytes modulo (2**31-1); independent of prompt and checkpoint",
        },
        "prompt_registry_sha256": json_hash(registry),
        "prompt_ids": [prompt.id for prompt in PROMPTS],
        "training_budget_comparison_prompt": "seen_1",
        "prompt_validation_rule": "At the final checkpoint, select among seven frozen prompts on the 72 held-out validation instances only: most feasible candidates, then lowest conditional mean gap, then registry order. Freeze this choice before test outcomes.",
        "prompt_test_rule": "Evaluate all seven frozen phrasings and the validation-selected choice using the same 216 test instances and ten candidates per instance.",
        "historical_protocol_caveats": {
            "historical_577_manifest_recovered": False,
            "new_test_subset_is_historical_577": False,
            "new_test_description": "Outcome-blind stratified 216-instance subset of the 1800-instance public release; a fresh controlled evaluation.",
            "historical_saved_phi3_checkpoint": "Recovered checkpoint-1750; rank128, alpha256, one epoch, 1799 planned optimizer steps.",
            "historical_saved_split": "Approximately115000 training/5000 validation inferred from 1799 planned steps and recorded evaluation throughput; exact historical split indices unavailable.",
            "fresh_split": "118000 training/2000 validation using the recorded seed42 preparation split; not an exact replay of historical checkpoint1750.",
            "published_training_script": "Rank128/alpha256 with118000/2000 split, but script specifies two epochs; this experiment follows the manuscript's one-epoch budget.",
        },
        "interpretation": "Single-seed training-duration analysis at25/50/75/100percent of a fresh complete one-epoch118000-example run; matched instance/prompt/candidate budgets. Conditional gaps must be reported with feasibility.",
        "artifact_sha256": artifact_hashes,
    }
    result["protocol_sha256"] = json_hash(result)
    (output_dir / "STUDY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "test_count", "prompt_validation_count",
                                                "checkpoint_steps", "protocol_sha256"]}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_study(args.prepared_manifest, args.prepared_test, args.output_dir)


if __name__ == "__main__":
    main()
