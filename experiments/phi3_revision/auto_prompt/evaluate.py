#!/usr/bin/env python3
"""Isolated dynamic-instruction evaluation using the unchanged frozen runner.

Only a private module copy's PROMPTS tuple is replaced. The shared core and
primary evaluation registries remain unchanged. Scientific datasets, sampling,
candidate counts, EOS handling, validation and resume follow the frozen runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.phi3_revision import core


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(path, key):
    value = json.loads(Path(path).read_text())
    if value.get(key) != _hash({name: item for name, item in value.items() if name != key}):
        raise ValueError(f"Invalid {key}: {path}")
    return value


def _read_protocol(path, verify_files=True):
    # The separate protocol module checks its own source and proposer bindings.
    from experiments.phi3_revision.auto_prompt.protocol import load_protocol
    return load_protocol(path, verify_files=verify_files)


def _validate_spec(spec, pilot=False):
    if not isinstance(spec, dict) or set(spec) != {"id", "text", "text_sha256", "origin", "token_count", "creation_order"}:
        raise ValueError("Instruction spec must contain the exact frozen fields")
    text = spec["text"]
    if (not isinstance(text, str) or not 20 <= len(text) <= 600 or text != text.strip()
            or "\n" in text or "\r" in text or "<|" in text or "|>" in text):
        raise ValueError("Instruction must be one clean 20--600 character line without role tokens")
    if spec["text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("Instruction text SHA256 mismatch")
    if type(spec["token_count"]) is not int or not 1 <= spec["token_count"] <= 128:
        raise ValueError("Instruction token count must be 1--128")
    origin = spec["origin"]
    if not isinstance(origin, dict) or set(origin) != {"round", "slot", "attempt"} or any(type(v) is not int for v in origin.values()):
        raise ValueError("Instruction origin is malformed")
    if pilot:
        if not re.fullmatch(r"auto_[A-Za-z0-9_]+", spec["id"]):
            raise ValueError("Invalid excluded pilot instruction ID")
        return
    if spec["id"] == "auto_0":
        if (origin != {"round": 0, "slot": 0, "attempt": 0} or spec["creation_order"] != 0
                or text != core.PROMPTS[0].text):
            raise ValueError("Initial instruction must be exactly the first training instruction")
    else:
        round_id, slot, attempt = origin["round"], origin["slot"], origin["attempt"]
        if (round_id not in (1, 2) or slot not in (1, 2) or attempt not in range(1, 5)
                or spec["id"] != f"auto_r{round_id}_s{slot}"
                or spec["creation_order"] != 2 * (round_id - 1) + slot):
            raise ValueError("Generated instruction ID/origin/order differs from the frozen search")


@contextmanager
def _private_evaluator(spec):
    """Copy the frozen module; never mutate either shared PROMPTS binding."""
    registries = [(module, module.PROMPTS) for module in list(sys.modules.values())
                  if module is not None and hasattr(module, "PROMPTS")
                  and getattr(module, "__file__", None)
                  and Path(module.__file__).resolve() in {(ROOT / "core.py").resolve(), (ROOT / "evaluate.py").resolve()}]
    name = "experiments.phi3_revision._auto_eval_" + uuid.uuid4().hex
    module_spec = importlib.util.spec_from_file_location(name, ROOT / "evaluate.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
        seen_in_training = any(prompt.seen_in_training and prompt.text == spec["text"] for prompt in core.PROMPTS)
        module.PROMPTS = (core.Prompt(spec["id"], spec["text"], seen_in_training),)
        yield module
    finally:
        sys.modules.pop(name, None)
        if any(module.PROMPTS is not registry for module, registry in registries):
            raise RuntimeError("Dynamic evaluation altered a shared frozen prompt registry")


def _selection(protocol, spec, pilot=False):
    if pilot:
        return None
    path = Path(protocol["output_dir"]) / "SELECTED.json"
    selected = _self_hashed(path, "selection_sha256")
    if (selected.get("status") != "frozen_before_auto_test"
            or selected.get("protocol_sha256") != protocol["protocol_sha256"]
            or selected.get("selected_instruction") != spec
            or selected.get("selected_instruction_id") != spec["id"]
            or selected.get("text_sha256") != spec["text_sha256"]):
        raise ValueError("Test instruction differs from the frozen validation-only selection")
    return selected


def _context(spec, data, adapter, outdir, base_path, *, protocol_path, split, study_dir=None, pilot=False):
    _validate_spec(spec, pilot)
    if split not in ("validation", "test"):
        raise ValueError("Split must be validation or test")
    protocol = _read_protocol(protocol_path, verify_files=not pilot)
    study_path = Path(protocol["parent_study"]).resolve()
    if study_dir is not None and Path(study_dir).resolve() != study_path.parent:
        raise ValueError("Explicit parent study differs from the auto protocol")
    study = _self_hashed(study_path, "protocol_sha256")
    if study["protocol_sha256"] != protocol["parent_protocol_sha256"]:
        raise ValueError("Auto protocol belongs to a different parent study")
    data, adapter, outdir, base_path = [Path(path).resolve() for path in (data, adapter, outdir, base_path)]
    if base_path != Path(protocol["base_path"]).resolve():
        raise ValueError("Dynamic evaluator uses a different base model")
    selection = _selection(protocol, spec, pilot) if split == "test" else None
    if not pilot:
        if data != Path(protocol["validation_file" if split == "validation" else "test_file"]).resolve():
            raise ValueError("Dynamic evaluation may only use its exact parent split")
        filename = "prompt_validation.jsonl" if split == "validation" else "test.jsonl"
        if _file_hash(data) != study["artifact_sha256"][filename]:
            raise ValueError("Parent evaluation split bytes changed")
        if adapter != Path(protocol["adapter_path"]).resolve():
            raise ValueError("Dynamic evaluator must use the final trained adapter")
        marker = json.loads((adapter / "experiment_checkpoint.json").read_text())
        completion = json.loads((adapter.parent / "training_complete.json").read_text())
        if marker.get("global_step") != 1844 or completion.get("global_step") != 1844:
            raise ValueError("Dynamic evaluation requires the completed final checkpoint")
        expected_out = Path(protocol["output_dir"]).resolve() / "evaluations" / split / spec["id"]
        if outdir != expected_out:
            raise ValueError("Instruction output is outside its frozen slot")
    with _private_evaluator(spec) as module:
        items, total = module.load_dataset(data)
        identities = [item.instance_id for item in items]
        expected_ids = protocol["validation_instance_ids" if split == "validation" else "test_instance_ids"]
        if pilot:
            if len(items) != 1 or set(identities) & set(protocol["validation_instance_ids"] + protocol["test_instance_ids"]):
                raise ValueError("Engineering pilot must use one toy instance excluded from both parent populations")
        elif identities != expected_ids or len(items) != (72 if split == "validation" else 216):
            raise ValueError("Dynamic evaluation canonical population differs from the frozen split")
        config = module.EvaluationConfig(
            str(base_path), str(data), str(outdir), adapter=str(adapter), prompt_ids=(spec["id"],),
            candidate_count=3 if split == "validation" else 10,
            batch_size=3 if split == "validation" else 5, seed=42, max_new_tokens=None,
            limit=1 if pilot else None, input_field="prompt_machines_first", temperature=1.0, top_k=50, top_p=.95)
    return protocol, config, items, selection


def _instruction_manifest(spec, config, protocol, native_manifest, native_path, selection, pilot, token_verified):
    return {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
            "split": "validation" if config.candidate_count == 3 else "test", "instruction": spec,
            "instruction_text_sha256": spec["text_sha256"], "parent_study": protocol["parent_study"],
            "parent_protocol_sha256": protocol["parent_protocol_sha256"],
            "data_path": config.data, "data_sha256": _file_hash(config.data),
            "base_path": config.base_path, "adapter_path": config.adapter,
            "native_configuration_sha256": native_manifest["configuration_sha256"],
            "native_evaluation_manifest_sha256": _file_hash(native_path),
            "selected_before_test_sha256": selection["selection_sha256"] if selection else None,
            "wrapper_source_sha256": _file_hash(__file__), "pilot_excluded": pilot,
            "seen_in_parent_training": any(prompt.seen_in_training and prompt.text == spec["text"] for prompt in core.PROMPTS),
            "exact_parent_prompt_matches": [prompt.id for prompt in core.PROMPTS if prompt.text == spec["text"]],
            "official_token_count_verified": token_verified}


def _run_in_process(spec, data, adapter, outdir, base_path, *, protocol_path, split, study_dir=None, pilot=False, engine=None):
    """Child entry; fake engines are restricted to explicitly excluded pilots."""
    if engine is not None and not pilot:
        raise ValueError("Fake evaluation engines are allowed only for excluded engineering pilots")
    protocol, config, items, selection = _context(spec, data, adapter, outdir, base_path,
        protocol_path=protocol_path, split=split, study_dir=study_dir, pilot=pilot)
    if engine is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_path, local_files_only=True, trust_remote_code=False)
        actual_tokens = len(tokenizer.encode(spec["text"], add_special_tokens=False))
        if actual_tokens != spec["token_count"]:
            raise ValueError("Instruction token_count disagrees with the official base tokenizer")
    with _private_evaluator(spec) as module:
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        native = module.build_manifest(config, items, len(items))
        native_path = output / "evaluation_manifest.json"
        if native_path.exists():
            if json.loads(native_path.read_text()) != native:
                raise ValueError("Resume refused: dynamic registry or frozen native configuration changed")
        else:
            if any(output.iterdir()):
                raise ValueError("Nonempty dynamic instruction slot has no native manifest")
            module.atomic_json(native_path, native)
        manifest_path = output / "instruction_manifest.json"
        metadata = _instruction_manifest(spec, config, protocol, native, native_path, selection, pilot, engine is None)
        if manifest_path.exists():
            saved = _self_hashed(manifest_path, "manifest_sha256")
            if {k: v for k, v in saved.items() if k not in ("created_at_utc", "manifest_sha256")} != metadata:
                raise ValueError("Resume refused: generated instruction or auto protocol binding changed")
        else:
            metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["manifest_sha256"] = _hash(metadata)
            module.atomic_json(manifest_path, metadata)
        return module.run(config, engine=engine)


def run_instruction(spec, data, adapter, outdir, base_path, *, protocol_path, split, study_dir=None,
                    python=None, device_uuid=None, pilot=False):
    """Launch one private native-evaluator process, then verify every raw slot."""
    _context(spec, data, adapter, outdir, base_path, protocol_path=protocol_path, split=split, study_dir=study_dir, pilot=pilot)
    arguments = {"spec": spec, "data": str(data), "adapter": str(adapter), "outdir": str(outdir),
                 "base_path": str(base_path), "protocol_path": str(protocol_path), "split": split,
                 "study_dir": str(study_dir) if study_dir else None, "pilot": pilot}
    environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if device_uuid is not None:
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device_uuid):
            raise ValueError("A dynamic evaluator device must be an explicit GPU UUID")
        environment["CUDA_VISIBLE_DEVICES"] = device_uuid
    result = subprocess.run([python or sys.executable, "-B", str(Path(__file__).resolve()), "--run-child"],
                            input=json.dumps(arguments), text=True, capture_output=True, env=environment)
    if Path(outdir).is_dir():
        (Path(outdir) / "child_output.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Isolated dynamic instruction evaluation failed: " + result.stderr[-2000:])
    return verify_instruction(spec, data, adapter, outdir, base_path, protocol_path=protocol_path,
                              split=split, study_dir=study_dir, pilot=pilot)


def verify_instruction(spec, data, adapter, outdir, base_path, *, protocol_path, split, study_dir=None, pilot=False):
    """Read-only verification under the exact dynamic registry; never generate."""
    protocol, config, items, selection = _context(spec, data, adapter, outdir, base_path,
        protocol_path=protocol_path, split=split, study_dir=study_dir, pilot=pilot)
    output = Path(outdir)
    with _private_evaluator(spec) as module:
        native_path = output / "evaluation_manifest.json"
        native = _self_hashed(native_path, "configuration_sha256")
        marker = json.loads((output / "COMPLETE").read_text())
        expected = module.build_manifest(config, items, len(items))
        # Verification may run under the reporting interpreter. Resuming native
        # generation still checks exact package versions inside the child run.
        expected["packages"] = native["packages"]
        expected.pop("configuration_sha256")
        expected["configuration_sha256"] = module.json_hash(expected)
        if expected != native:
            raise ValueError("Dynamic evaluation registry, source, model, split or decoding changed")
        metadata = _self_hashed(output / "instruction_manifest.json", "manifest_sha256")
        stable = _instruction_manifest(spec, config, protocol, native, native_path, selection, pilot, metadata["official_token_count_verified"])
        if {k: v for k, v in metadata.items() if k not in ("created_at_utc", "manifest_sha256")} != stable:
            raise ValueError("Dynamic instruction manifest does not match the generated instruction and protocol")
        if not pilot and not metadata["official_token_count_verified"]:
            raise ValueError("Scientific instruction token count was not verified with the official tokenizer")
        if split == "test" and not pilot:
            selection_path = Path(protocol["output_dir"]) / "SELECTED.json"
            if selection_path.stat().st_mtime > native_path.stat().st_mtime:
                raise ValueError("Final instruction selection was modified after auto test began")
        if marker.get("status") != "complete" or marker.get("configuration_sha256") != native["configuration_sha256"]:
            raise ValueError("Dynamic evaluation is incomplete or contains context skips")
        progress = module.load_progress(output / "candidates.jsonl", items, config, native["configuration_sha256"])
        summary = module.summarize(progress, items, config, native)
        expected_slots = len(items) * config.candidate_count
        if (summary["status"] != "complete" or len(progress) != expected_slots
                or marker.get("candidate_slots") != expected_slots):
            raise ValueError("Dynamic evaluation does not contain every fixed candidate slot")
        for row in progress.values():
            item = next(item for item in items if item.instance_id == row["instance_id"])
            if (row["max_new_tokens"] != module.token_budget(item, config)
                    or type(row.get("output_tokens")) is not int or not 0 <= row["output_tokens"] <= row["max_new_tokens"]
                    or type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0
                    or type(row.get("hit_max_new_tokens")) is not bool
                    or (row["hit_max_new_tokens"] and row["output_tokens"] != row["max_new_tokens"])):
                raise ValueError("Dynamic raw output violated its fixed generation budget")
        runtime_path = output / "runtime_device.json"
        if not pilot:
            runtime = json.loads(runtime_path.read_text())
            generation = json.loads((Path(base_path) / "generation_config.json").read_text())
            stops = generation.get("eos_token_id")
            stops = stops if isinstance(stops, list) else [stops]
            if (runtime.get("dtype") != "float16" or runtime.get("attention_implementation") != "sdpa"
                    or runtime.get("quantized") is not False or runtime.get("eos_token_ids") != stops
                    or runtime.get("torch_version") != native["packages"]["torch"]):
                raise ValueError("Dynamic runtime does not match native precision, EOS or package provenance")
        records, raw = [], list(progress.values())
        by_id = {item.instance_id: item.instance for item in items}
        for (identity, _, candidate_id), row in progress.items():
            records.append(core.EvaluationRecord(identity, candidate_id, core.validate_schedule(by_id[identity], row["output"])))
        metric = summary["by_prompt"][spec["id"]]
        candidates_hash = _file_hash(output / "candidates.jsonl")
        feedback = {"instruction_id": spec["id"], "instruction_sha256": spec["text_sha256"],
                    "instance_ids_sha256": _hash([item.instance_id for item in items]),
                    "n_instances": len(items), "candidate_count": config.candidate_count,
                    "n_candidates": metric["n_candidates"], "feasible_candidates": metric["feasible_candidates"],
                    "conditional_mean_gap_percent": metric["conditional_candidate_gap"]["mean_gap_percent"],
                    "error_counts": metric["error_counts_candidate_denominator"],
                    "evaluation_configuration_sha256": native["configuration_sha256"],
                    "raw_candidates_sha256": candidates_hash, "split": split,
                    "protocol_sha256": protocol["protocol_sha256"], "pilot_excluded": pilot}
        receipt = {"evaluation_manifest_sha256": _file_hash(native_path), "candidates_sha256": candidates_hash,
                   "complete_sha256": _file_hash(output / "COMPLETE"),
                   "instruction_manifest_sha256": _file_hash(output / "instruction_manifest.json")}
        return {"feedback": feedback, "receipt": receipt, "summary": summary, "records": records,
                "manifest": native, "instruction_manifest": metadata, "raw": raw}


def validate_dynamic_evaluation(folder, data_path, candidate_count, instruction_specs, protocol):
    """Reporter-compatible tuple for one completed, explicitly bound instruction."""
    metadata = _self_hashed(Path(folder) / "instruction_manifest.json", "manifest_sha256")
    spec = metadata["instruction"]
    specs = list(instruction_specs.values()) if isinstance(instruction_specs, dict) else list(instruction_specs)
    if spec not in specs or metadata["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Reporter instruction registry differs from the evaluated instruction")
    split = "validation" if candidate_count == 3 else "test" if candidate_count == 10 else None
    if split is None or metadata["split"] != split or metadata["pilot_excluded"]:
        raise ValueError("Reporter refuses pilot or non-frozen candidate budgets")
    protocol_path = Path(protocol.get("_protocol_path", Path(protocol["output_dir"]).parent / "PROTOCOL.json"))
    result = verify_instruction(spec, data_path, protocol["adapter_path"], folder, protocol["base_path"],
                                 protocol_path=protocol_path, split=split,
                                 study_dir=Path(protocol["parent_study"]).parent)
    return result["manifest"], result["summary"], {spec["id"]: result["records"]}, result["raw"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-child", action="store_true", required=True)
    parser.parse_args()
    arguments = json.loads(sys.stdin.read())
    _run_in_process(**arguments)


if __name__ == "__main__":
    main()
