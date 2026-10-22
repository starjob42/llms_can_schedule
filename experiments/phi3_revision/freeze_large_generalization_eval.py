#!/usr/bin/env python3
"""Freeze a held-out larger-instance generalization population.

This diagnostic is deliberately separate from the frozen 216-instance test
study. It samples only records in the recorded 2,000-example validation
partition, stratified by larger job/machine shape, and never uses outcomes to
choose records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .core import canonical_hash, parse_instance
except ImportError:
    from core import canonical_hash, parse_instance


DEFAULT_SHAPES = (
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8),
    (10, 9), (10, 10), (12, 12), (15, 10), (15, 15), (20, 15), (20, 20),
)


def digest_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(shape: tuple[int, int], instance_id: str, seed: int) -> str:
    jobs, machines = shape
    return hashlib.sha256(
        f"phi3_large_generalization:{seed}:{jobs}x{machines}:{instance_id}".encode()
    ).hexdigest()


def freeze(
    source: Path,
    split: Path,
    output: Path,
    manifest_path: Path,
    per_shape: int,
    seed: int,
    allow_training_source: bool,
) -> dict:
    if per_shape <= 0:
        raise ValueError("per_shape must be positive")
    rows = json.loads(source.read_text())
    split_value = json.loads(split.read_text())
    validation_indices = split_value["validation_indices"]
    validation_set = set(validation_indices)
    candidates: dict[tuple[int, int], list[dict]] = {shape: [] for shape in DEFAULT_SHAPES}
    candidate_indices = range(len(rows)) if allow_training_source else validation_indices
    for source_index in candidate_indices:
        row = rows[source_index]
        shape = (int(row["num_jobs"]), int(row["num_machines"]))
        if shape not in candidates:
            continue
        instance_id = canonical_hash(parse_instance(row["prompt_machines_first"]))
        candidates[shape].append({
            "source_index": source_index,
            "instance_id": instance_id,
            "row": row,
        })
    if any(len(values) < per_shape for values in candidates.values()):
        raise ValueError("Validation partition lacks enough records for a requested shape")
    selected = []
    for shape in DEFAULT_SHAPES:
        values = sorted(
            candidates[shape],
            key=lambda item: rank(shape, item["instance_id"], seed),
        )[:per_shape]
        for item in values:
            row = dict(item["row"])
            row.update({
                "instance_id": item["instance_id"],
                "source_index": item["source_index"],
                "source_split": (
                    "training_source_validation"
                    if item["source_index"] in validation_set
                    else "training_source_training"
                ),
                "selection_rank": rank(shape, item["instance_id"], seed),
            })
            selected.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_model_outcomes",
        "purpose": "LARGER_INSTANCE_GENERALIZATION_DIAGNOSTIC",
        "scientific_warning": (
            "This population includes training-source records because "
            "allow_training_source was explicitly selected; it is not a "
            "held-out generalization evaluation."
            if allow_training_source else
            "This population contains only records from the frozen validation partition."
        ),
        "source": str(source.resolve()),
        "source_sha256": digest_bytes(source),
        "split": str(split.resolve()),
        "split_sha256": digest_bytes(split),
        "source_population": len(rows),
        "validation_population": len(validation_indices),
        "allow_training_source": allow_training_source,
        "validation_indices_are_unique": len(validation_set) == len(validation_indices),
        "selection_seed": seed,
        "selection_rule": "Within each larger shape, select the lowest SHA256(phi3_large_generalization:seed:shape:instance_id) ranks; outcomes and reference labels are not consulted.",
        "shapes": [f"{jobs}x{machines}" for jobs, machines in DEFAULT_SHAPES],
        "per_shape": per_shape,
        "evaluated_instances": len(selected),
        "candidate_count": 10,
        "prompt_ids": ["seen_1"],
        "checkpoint": "main/checkpoint-1844",
        "input_field": "prompt_machines_first",
        "data_path": str(output.resolve()),
        "data_sha256": digest_bytes(output),
        "source_indices": [row["source_index"] for row in selected],
        "instance_ids": [row["instance_id"] for row in selected],
        "shape_counts": {
            f"{jobs}x{machines}": sum(
                row["num_jobs"] == jobs and row["num_machines"] == machines
                for row in selected
            )
            for jobs, machines in DEFAULT_SHAPES
        },
    }
    manifest["expected_candidate_slots"] = len(selected) * manifest["candidate_count"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=root / "original_release/jssp_llm_format_120k.json")
    parser.add_argument("--split", type=Path, default=root / "study/PREPARATION_SPLIT.json")
    parser.add_argument("--output", type=Path, default=root / "study/large_generalization_26_per_shape.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "study/LARGE_GENERALIZATION_HELDOUT_EVAL.json")
    parser.add_argument("--per-shape", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-training-source", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze(
        args.source, args.split, args.output, args.manifest, args.per_shape,
        args.seed, args.allow_training_source,
    ), indent=2))


if __name__ == "__main__":
    main()
