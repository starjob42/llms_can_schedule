"""Audit authentic JSSP records and freeze a leakage-free Phi-3 experiment.

No model is trained here. Original files are read-only; all derived artifacts
are written beneath --output-dir. Matrix content, rather than paths or prompt
strings, determines identity.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from core import canonical_hash, parse_instance, parse_matrix


def matrix_identity(record):
    instance = parse_matrix(record["matrix"]) if record.get("matrix") else parse_instance(record["prompt_jobs_first"])
    return canonical_hash(instance)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "x") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-source", type=Path, required=True)
    p.add_argument("--test-source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = json.loads(args.training_source.read_text())
    test = json.loads(args.test_source.read_text())
    train_ids = [matrix_identity(r) for r in train]
    test_ids = [matrix_identity(r) for r in test]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate source instances: explicit deduplication required")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Train/test contamination: {len(overlap)} shared matrices")
    if not 0 < args.validation_size < len(train):
        raise ValueError("Invalid validation split size")
    # This is a new, explicitly recorded split; it is not claimed to reproduce
    # a historical split whose manifest has not been recovered.
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    val_indices, train_indices = order[:args.validation_size], order[args.validation_size:]
    dimensions = collections.Counter()
    for r in test:
        instance = parse_matrix(r["matrix"]) if r.get("matrix") else parse_instance(r["prompt_jobs_first"])
        dimensions[f"{instance.n_jobs}x{instance.n_machines}"] += 1
    manifest = {
        "status": "data_audit_complete",
        "training_source": str(args.training_source.resolve()),
        "training_source_sha256": sha256_file(args.training_source),
        "test_source": str(args.test_source.resolve()),
        "test_source_sha256": sha256_file(args.test_source),
        "training_count": len(train_indices), "validation_count": len(val_indices),
        "test_count": len(test), "unique_training_source": len(set(train_ids)),
        "unique_test": len(set(test_ids)), "train_test_overlap": len(overlap),
        "test_dimensions": dict(sorted(dimensions.items())), "seed": args.seed,
        "split_algorithm": "python_random_seeded_shuffle_validation_prefix",
        "historical_577_manifest_recovered": False,
        "representation": "source_machine_first_preserved_verbatim",
        "training_indices": train_indices, "validation_indices": val_indices,
        "test_instance_ids": test_ids,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_jsonl(args.output_dir / "test.jsonl", [dict(r, instance_id=h) for r,h in zip(test,test_ids)])
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('indices') and k!='test_instance_ids'}, indent=2), flush=True)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        prompts = [
            "Instruct: Provide a solution schedule for the JSSP problem below, also indicate the makespan.",
            "Task: Provide the steps of a solution for the JSSP problem and determine the makespan.",
            "Command: Give a detailed solution to tackle the JSSP problem, focusing on optimizing the makespan.",
        ]
        rng = random.Random(args.seed)
        token_counts = {}
        for name, indices in [("train",train_indices),("validation",val_indices)]:
            lengths = []
            target = args.output_dir / f"{name}.jsonl"
            with target.open("x") as f:
                for n, idx in enumerate(indices):
                    row = train[idx]
                    messages = [
                        {"role":"system","content":"You are expert in Job Shop Scheduling Problem"},
                        {"role":"user","content":f" {rng.choice(prompts)} \n{row['prompt_machines_first']} "},
                        {"role":"assistant","content":f" {row['output']} "},
                    ]
                    ids = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
                    if len(ids)>40000:
                        raise ValueError(f"Refusing to truncate {name} row {idx}: {len(ids)} tokens")
                    lengths.append(len(ids))
                    f.write(json.dumps({"input_ids":ids,"instance_id":train_ids[idx],"source_index":idx},separators=(",", ":"))+"\n")
                    if n%5000==0:
                        print(f"tokenized {name} {n}/{len(indices)}",flush=True)
            token_counts[name] = {"count":len(lengths),"total":sum(lengths),"maximum":max(lengths),"mean":sum(lengths)/len(lengths),"sha256":sha256_file(target)}
        manifest["tokenization"] = token_counts
        manifest["tokenizer_path"] = str(args.tokenizer.resolve())
        manifest["tokenizer_config_sha256"] = sha256_file(args.tokenizer / "tokenizer_config.json")
        manifest["status"] = "prepared"
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps(token_counts,indent=2),flush=True)


if __name__ == "__main__":
    main()
