#!/usr/bin/env python3
"""Prepare the independent Taillard/DMU JSSP evaluation population.

The shared benchmark sources contain ten instances per group. This prepares all
groups with no dimension exceeding 50 jobs by 20 machines, excluding Taillard
100x20. Reference schedules are converted to the strict project format and
validated before the frozen JSONL is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .core import canonical_hash, parse_matrix, validate_schedule
except ImportError:
    from core import canonical_hash, parse_matrix, validate_schedule


DEFAULT_SOURCES = {
    "taillard": Path("/home/lamsade/ksusini/ACCORD/validation_data/jssp_benchmark_tai_data.json"),
    "dmu": Path("/home/lamsade/ksusini/ACCORD/validation_data/jssp_benchmark_dmu_data.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_text(row: dict) -> str:
    instance = parse_matrix(row["matrix"])
    serialized = row["output_list_of_lists"].splitlines()[0]
    timings = json.loads(serialized)
    if not isinstance(timings, list):
        raise ValueError("Benchmark output_list_of_lists must be a list")
    by_key = {(item[0], item[1]): item for item in timings}
    lines = ["Solution:"]
    for operation in instance.operations:
        item = by_key.get((operation.job, operation.machine))
        if item is None or len(item) != 4:
            raise ValueError(f"Missing benchmark reference operation {(operation.job, operation.machine)}")
        _, _, start, duration = item
        end = float(start) + float(duration)
        lines.append(
            f"Job {operation.job} Operation {operation.operation} on Machine "
            f"{operation.machine} : {start} + {duration} -> {end:g}"
        )
    lines.append(f"Makespan: {float(row['makespan']):g}")
    return "\n".join(lines) + "\n"


def prepare(sources: dict[str, Path], output: Path, manifest_path: Path) -> dict:
    rows = []
    source_hashes = {}
    for family, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hashes[family] = {"path": str(source), "sha256": sha256(source)}
        for row in json.loads(source.read_text()):
            jobs, machines = int(row["num_jobs"]), int(row["num_machines"])
            if jobs > 50 or machines > 20:
                continue
            instance = parse_matrix(row["matrix"])
            reference = reference_text(row)
            result = validate_schedule(instance, reference)
            if not result.feasible:
                raise ValueError(f"Invalid {family} reference {row['path']}: {result.errors}")
            rows.append({
                "benchmark_family": family,
                "benchmark_path": row["path"],
                "instance_id": canonical_hash(instance),
                "num_jobs": jobs,
                "num_machines": machines,
                "matrix": row["matrix"],
                "prompt_machines_first": row["input"],
                "prompt_jobs_first": row["input"],
                "reference_schedule": reference,
                "reference_makespan": float(row["makespan"]),
                "source_split": "external_heldout_benchmark",
            })
    rows.sort(key=lambda row: (
        row["num_jobs"] * row["num_machines"],
        row["num_jobs"],
        row["num_machines"],
        row["benchmark_family"],
        row["benchmark_path"],
    ))
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical benchmark instance")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    counts = {}
    for row in rows:
        key = f"{row['benchmark_family']}:{row['num_jobs']}x{row['num_machines']}"
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_model_outcomes",
        "purpose": "EXTERNAL_HELDOUT_TAILLARD_DMU_DIAGNOSTIC",
        "dimension_cap": "50x20",
        "excluded_groups": ["taillard:100x20"],
        "sources": source_hashes,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "evaluated_instances": len(rows),
        "candidate_count": 10,
        "prompt_ids": ["seen_1"],
        "checkpoint": "main/checkpoint-1844",
        "shape_counts": counts,
        "instance_ids": ids,
        "reference_labels": "Imported benchmark schedules; feasibility validated by the project validator; benchmark makespans are not asserted optimal by this protocol.",
    }
    manifest["expected_candidate_slots"] = len(rows) * manifest["candidate_count"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "study/taillard_dmu_50x20.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "study/TAILLARD_DMU_50X20_EVAL.json")
    args = parser.parse_args()
    print(json.dumps(prepare(DEFAULT_SOURCES, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
