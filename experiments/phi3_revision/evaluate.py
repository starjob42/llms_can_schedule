#!/usr/bin/env python3
"""Resumable paired Phi-3 evaluation; imports do not load a model or use a GPU.

All candidate slots are fixed in advance. Failed schedules retain their place
in feasibility denominators; gap summaries condition on feasible schedules.
Limited runs are prominently marked PILOT and cannot masquerade as full runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Sequence

try:
    from .core import (EvaluationRecord, Instance, PROMPTS, SYSTEM_PROMPT,
                       ValidationResult, aggregate_results, canonical_hash,
                       parse_instance, parse_matrix, render_user_prompt,
                       validate_schedule)
except ImportError:
    from core import (EvaluationRecord, Instance, PROMPTS, SYSTEM_PROMPT,
                      ValidationResult, aggregate_results, canonical_hash,
                      parse_instance, parse_matrix, render_user_prompt,
                      validate_schedule)


EVALUATION_SYSTEM_PROMPT = SYSTEM_PROMPT


def evaluation_user_prompt(prompt_id: str, input_text: str) -> str:
    """Preserve the recovered original inference utility's exact whitespace."""
    prompt = next((prompt for prompt in PROMPTS if prompt.id == prompt_id), None)
    if prompt is None:
        raise ValueError(f"Unknown frozen prompt: {prompt_id}")
    return f"{prompt.text}\n{input_text}"


def eos_token_ids(configured, tokenizer_eos: int | None) -> tuple[int, ...]:
    """Honor all official generation stops, including Phi-3's chat end token."""
    values = configured if configured is not None else tokenizer_eos
    values = values if isinstance(values, (tuple, list)) else [values]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Model must define valid EOS token IDs")
    return tuple(dict.fromkeys(values))


def trim_generated_tokens(ids: list[int], stops: Sequence[int]) -> list[int]:
    for index, token in enumerate(ids):
        if token in stops:
            return ids[:index + 1]
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def artifact_digest(path: Path, *, adapter: bool = False) -> dict:
    """Hash inference artifacts, excluding optimizer/training state and logs."""
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Model directory does not exist: {path}")
    adapter_names = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    excluded = {"optimizer.pt", "scheduler.pt", "training_args.bin", "trainer_state.json"}
    files = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.name in excluded or file.name.startswith("rng_state"):
            continue
        if adapter:
            relevant = file.name in adapter_names
        else:
            relevant = file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".py", ".tiktoken"}
        if relevant:
            files.append({"file": str(file.relative_to(path)), "size": file.stat().st_size,
                          "sha256": sha256_file(file)})
    if adapter and not any(item["file"] in {"adapter_model.safetensors", "adapter_model.bin"} for item in files):
        raise ValueError(f"No adapter weights found: {path}")
    if not adapter and not any(item["file"].endswith((".safetensors", ".bin")) for item in files):
        raise ValueError(f"No base model weights found: {path}")
    return {"path": str(path), "sha256": json_hash(files), "files": files}


@dataclass(frozen=True)
class EvaluationItem:
    instance_id: str
    instance: Instance
    input_text: str
    source_index: int
    source_instance_id: str | None
    input_encodes_precedence: bool


def _compact_machine_audit(text: str, instance: Instance) -> None:
    observed = []
    machine = None
    for line in text.splitlines():
        header = re.fullmatch(r"\s*M(\d+)\s*:\s*", line)
        if header:
            machine = int(header[1])
            continue
        if machine is not None and line.strip():
            pairs = list(re.finditer(r"J(\d+)\s*:\s*(\d+(?:\.\d+)?)", line))
            rest = re.sub(r"J\d+\s*:\s*\d+(?:\.\d+)?", "", line).strip()
            if rest or not pairs:
                raise ValueError("Malformed compact machine-first row")
            observed.extend((int(pair[1]), machine, float(pair[2])) for pair in pairs)
    expected = [(op.job, op.machine, float(op.duration)) for op in instance.operations]
    if Counter(observed) != Counter(expected):
        raise ValueError("Compact machine-first input disagrees with the matrix")


def load_dataset(path: Path, input_field: str = "prompt_machines_first", limit: int | None = None) -> tuple[list[EvaluationItem], int]:
    text = path.read_text()
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation data must contain a nonempty list of records")
    if limit is not None and limit <= 0:
        raise ValueError("Pilot limit must be positive")
    total_source = len(rows)
    # Audit every source identity, including rows outside an explicitly limited pilot.
    items = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        instance = parse_matrix(row["matrix"]) if row.get("matrix") else parse_instance(row.get("prompt_jobs_first", row.get("instruction", "")))
        identity = canonical_hash(instance)
        if identity in seen:
            raise ValueError(f"Duplicate canonical instance at row {index}: {identity}")
        seen.add(identity)
        input_text = row.get(input_field)
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"Record {index} lacks requested input field {input_field}")
        precedence_encoded = not bool(re.search(r"(?m)^\s*M\d+\s*:\s*$", input_text))
        if precedence_encoded:
            if canonical_hash(parse_instance(input_text)) != identity:
                raise ValueError(f"Input/matrix mismatch in record {index}")
        else:
            _compact_machine_audit(input_text, instance)
        if row.get("output"):
            gold = validate_schedule(instance, row["output"])
            if not gold.feasible:
                raise ValueError(f"Invalid solver label in record {index}: {gold.errors}")
            if instance.reference_makespan is None:
                instance = replace(instance, reference_makespan=gold.computed_makespan)
            elif not math.isclose(instance.reference_makespan, gold.computed_makespan, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Reference/solver-label mismatch in record {index}")
        if limit is None or index < limit:
            items.append(EvaluationItem(identity, instance, input_text, index, row.get("instance_id"), precedence_encoded))
    return items, total_source


@dataclass(frozen=True)
class EvaluationConfig:
    base_path: str
    data: str
    output_dir: str
    adapter: str | None = None
    prompt_ids: tuple[str, ...] = ("seen_1",)
    candidate_count: int = 10
    batch_size: int = 5
    seed: int = 42
    max_new_tokens: int | None = None
    limit: int | None = None
    input_field: str = "prompt_machines_first"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    def validate(self) -> None:
        if self.candidate_count < 1 or self.batch_size < 1:
            raise ValueError("Candidate count and batch size must be positive")
        known = {prompt.id for prompt in PROMPTS}
        if not self.prompt_ids or len(set(self.prompt_ids)) != len(self.prompt_ids) or not set(self.prompt_ids) <= known:
            raise ValueError("Prompt IDs must be unique frozen registry IDs")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("Pilot limit must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Require finite positive temperature and top_p in (0,1]")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")


def generation_seed(instance_id: str, seed: int, batch_index: int) -> int:
    # Prompt ID and model checkpoint are intentionally absent for pairing.
    digest = hashlib.sha256(f"{instance_id}:{seed}:{batch_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def token_budget(item: EvaluationItem, config: EvaluationConfig) -> int:
    return config.max_new_tokens or min(8192, len(item.instance.operations) * 48 + 128)


def build_manifest(config: EvaluationConfig, items: Sequence[EvaluationItem], total_source: int) -> dict:
    config.validate()
    effective = asdict(config)
    effective.pop("output_dir")
    effective["prompt_ids"] = list(config.prompt_ids)
    effective["base_path"] = str(Path(config.base_path).resolve())
    effective["data"] = str(Path(config.data).resolve())
    if config.adapter:
        effective["adapter"] = str(Path(config.adapter).resolve())
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not_installed"
    source = Path(__file__).resolve()
    manifest = {
        "schema_version": 1, "protocol": "phi3_fp16_sdpa_no_quantization",
        "run_kind": "PILOT_LIMITED" if config.limit is not None else "FULL",
        "config": effective, "source_data_sha256": sha256_file(Path(config.data)),
        "base_artifacts": artifact_digest(Path(config.base_path)),
        "adapter_artifacts": artifact_digest(Path(config.adapter), adapter=True) if config.adapter else None,
        "source_code_sha256": {source.name: sha256_file(source), "core.py": sha256_file(source.with_name("core.py"))},
        "packages": packages, "frozen_prompts": [asdict(prompt) for prompt in PROMPTS],
        "system_prompt": EVALUATION_SYSTEM_PROMPT,
        "user_template": "{instruction}\n{problem}",
        "chat_framing": "native tokenizer add_generation_prompt=True (released utility used False)",
        "generation_stop_rule": "all base generation_config.eos_token_id values; tokenizer eos only if unspecified",
        "source_population": total_source,
        "evaluated_instances": len(items), "instance_ids": [item.instance_id for item in items],
        "expected_candidate_slots": len(items) * len(config.prompt_ids) * config.candidate_count,
        "input_encodes_precedence": all(item.input_encodes_precedence for item in items),
        "generation_seed_rule": "sha256(instance_id:seed:batch_index) first 8 bytes mod (2**31-1)",
        "generation_budget_rule": "explicit max_new_tokens or min(8192,48*n_operations+128)",
        "selection": "fixed candidate count; no validity-based stopping",
    }
    manifest["configuration_sha256"] = json_hash(manifest)
    return manifest


def skipped_validation(status: str) -> ValidationResult:
    return ValidationResult(False, (status,), ("No schedule generated: " + status,), (), None, None, None)


def load_progress(path: Path, items: Sequence[EvaluationItem], config: EvaluationConfig,
                  configuration_sha256: str | None = None) -> dict[tuple[str, str, int], dict]:
    """Revalidate raw outputs on resume; never trust saved feasibility fields."""
    if not path.exists():
        return {}
    by_id = {item.instance_id: item for item in items}
    progress = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Corrupt candidate JSONL line {line_number}; preserve and repair explicitly") from exc
            key = row.get("instance_id"), row.get("prompt_id"), row.get("candidate_id")
            if configuration_sha256 is not None and row.get("configuration_sha256") != configuration_sha256:
                raise ValueError("Persisted candidate belongs to another configuration")
            if key in progress:
                raise ValueError(f"Duplicate persisted candidate: {key}")
            instance_id, prompt_id, candidate_id = key
            if instance_id not in by_id or prompt_id not in config.prompt_ids or type(candidate_id) is not int or not 0 <= candidate_id < config.candidate_count:
                raise ValueError(f"Persisted candidate is outside the frozen experiment: {key}")
            expected_seed = generation_seed(instance_id, config.seed, candidate_id // config.batch_size)
            if row.get("generation_seed") != expected_seed:
                raise ValueError(f"Persisted candidate has the wrong paired seed: {key}")
            if row.get("status") == "generated":
                if not isinstance(row.get("output"), str):
                    raise ValueError("Persisted generated candidate lacks raw output")
                result = validate_schedule(by_id[instance_id].instance, row["output"])
            elif row.get("status") == "skipped_context":
                result = skipped_validation("skipped_context")
            else:
                raise ValueError(f"Unknown persisted candidate status: {row.get('status')}")
            row["validation"] = result.to_dict()
            progress[key] = row
    return progress


def summarize(progress: dict, items: Sequence[EvaluationItem], config: EvaluationConfig, manifest: dict) -> dict:
    records_by_prompt = defaultdict(list)
    statuses = Counter(row["status"] for row in progress.values())
    expected = manifest["expected_candidate_slots"]
    by_id = {item.instance_id: item for item in items}
    for (instance_id, prompt_id, candidate_id), row in progress.items():
        # Raw outputs were revalidated on resume and checked on generation.
        saved = row["validation"]
        result = ValidationResult(saved["feasible"], tuple(saved["errors"]), tuple(saved["details"]), (),
                                  saved["reported_makespan"], saved["computed_makespan"], saved["gap_percent"])
        records_by_prompt[prompt_id].append(EvaluationRecord(instance_id, candidate_id, result))
    # Include unattempted slots as explicit failures in population summaries.
    for item in items:
        for prompt_id in config.prompt_ids:
            for candidate_id in range(config.candidate_count):
                if (item.instance_id, prompt_id, candidate_id) not in progress:
                    records_by_prompt[prompt_id].append(EvaluationRecord(item.instance_id, candidate_id, skipped_validation("not_generated")))
    complete = len(progress) == expected
    return {
        "configuration_sha256": manifest["configuration_sha256"], "run_kind": manifest["run_kind"],
        "status": ("complete_with_skips" if statuses.get("skipped_context") else "complete") if complete else "incomplete",
        "source_population": manifest["source_population"], "n_evaluated_instances": len(items),
        "expected_candidate_slots": expected, "terminal_candidate_slots": len(progress),
        "not_generated_slots": expected - len(progress), "candidate_status_counts": dict(statuses),
        "input_encodes_precedence": manifest["input_encodes_precedence"],
        "denominator_note": "Per-prompt population includes skipped and ungenerated slots as failures. Gaps condition on feasible generated schedules. Candidate Wilson intervals assume independent Bernoulli trials; repeated candidates may be correlated.",
        "by_prompt": {prompt_id: aggregate_results(records_by_prompt[prompt_id], ks=tuple(sorted({1, 3, 5, config.candidate_count}))) for prompt_id in config.prompt_ids},
    }


class NativeGenerator:
    """Native transformers fp16 SDPA inference, with optional unmerged LoRA."""
    def __init__(self, config: EvaluationConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("GPU fp16 evaluation requires CUDA; no precision fallback is performed")
        self.torch = torch
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_path, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=False)
        if config.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, config.adapter, is_trainable=False, local_files_only=True)
        if getattr(self.model, "is_loaded_in_4bit", False) or getattr(self.model, "is_loaded_in_8bit", False):
            raise RuntimeError("Quantized inference violates the frozen evaluation protocol")
        self.model.eval()
        self.eos_token_ids = eos_token_ids(self.model.generation_config.eos_token_id, self.tokenizer.eos_token_id)
        self.max_context = int(self.model.config.max_position_embeddings)
        self.config = config

    def runtime_info(self) -> dict:
        properties = self.torch.cuda.get_device_properties(0)
        return {"hostname": socket.gethostname(), "gpu_name": properties.name,
                "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                "gpu_total_memory_bytes": properties.total_memory,
                "torch_version": self.torch.__version__, "dtype": "float16",
                "attention_implementation": "sdpa", "quantized": False,
                "eos_token_ids": list(self.eos_token_ids),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
                "currently_allocated_bytes": self.torch.cuda.memory_allocated(0)}

    def encode(self, user_text: str):
        messages = [{"role": "system", "content": EVALUATION_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return ids.to("cuda")

    def generate(self, encoded, count: int, seed: int, max_new_tokens: int) -> list[dict]:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=encoded, attention_mask=torch.ones_like(encoded),
                num_return_sequences=count, do_sample=True,
                temperature=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k,
                max_new_tokens=max_new_tokens, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=list(self.eos_token_ids))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        input_tokens = encoded.shape[-1]
        results = []
        for output in outputs:
            ids = trim_generated_tokens(output[input_tokens:].tolist(), self.eos_token_ids)
            results.append({"output": self.tokenizer.decode(ids, skip_special_tokens=True),
                            "input_tokens": input_tokens, "output_tokens": len(ids),
                            "batch_seconds": seconds, "amortized_seconds": seconds / count,
                            "hit_max_new_tokens": len(ids) >= max_new_tokens and (not ids or ids[-1] not in self.eos_token_ids)})
        return results


def _append_record(stream, row: dict) -> None:
    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(config: EvaluationConfig, *, engine=None) -> dict:
    started = time.monotonic()
    config.validate()
    items, total_source = load_dataset(Path(config.data), config.input_field, config.limit)
    manifest = build_manifest(config, items, total_source)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise ValueError("Resume refused: configuration, data, code, package versions, or model artifact hashes changed")
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Nonempty output directory has no matching evaluation manifest")
        atomic_json(manifest_path, manifest)
    progress_path = output_dir / "candidates.jsonl"
    progress = load_progress(progress_path, items, config, manifest["configuration_sha256"])
    # A previous COMPLETE marker never bypasses raw-output revalidation.
    summary = summarize(progress, items, config, manifest)
    if summary["status"].startswith("complete"):
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(output_dir / "COMPLETE", {"configuration_sha256": manifest["configuration_sha256"], "status": summary["status"], "candidate_slots": len(progress)})
        return summary
    complete_path = output_dir / "COMPLETE"
    if complete_path.exists():
        complete_path.unlink()
    atomic_json(output_dir / "summary.json", summary)
    last_summary_at = time.monotonic()
    try:
        engine = engine or NativeGenerator(config)
        if hasattr(engine, "runtime_info"):
            atomic_json(output_dir / "runtime_device.json", dict(engine.runtime_info(), elapsed_seconds=time.monotonic() - started))
        with progress_path.open("a") as stream:
            for item in items:
                for prompt_id in config.prompt_ids:
                    encoded = engine.encode(evaluation_user_prompt(prompt_id, item.input_text))
                    input_tokens = int(encoded.shape[-1])
                    budget = token_budget(item, config)
                    for batch_start in range(0, config.candidate_count, config.batch_size):
                        count = min(config.batch_size, config.candidate_count - batch_start)
                        candidate_ids = list(range(batch_start, batch_start + count))
                        if all((item.instance_id, prompt_id, candidate_id) in progress for candidate_id in candidate_ids):
                            continue
                        seed = generation_seed(item.instance_id, config.seed, batch_start // config.batch_size)
                        if input_tokens + budget > engine.max_context:
                            generated = [{"output": "", "input_tokens": input_tokens, "output_tokens": 0,
                                          "batch_seconds": 0.0, "amortized_seconds": 0.0,
                                          "hit_max_new_tokens": False, "status": "skipped_context"} for _ in candidate_ids]
                        else:
                            generated = engine.generate(encoded, count, seed, budget)
                            if len(generated) != count:
                                raise RuntimeError("Generator returned the wrong number of fixed candidates")
                        for candidate_id, sample in zip(candidate_ids, generated):
                            key = item.instance_id, prompt_id, candidate_id
                            if key in progress:
                                continue
                            status = sample.get("status", "generated")
                            result = validate_schedule(item.instance, sample["output"]) if status == "generated" else skipped_validation(status)
                            row = dict(sample, status=status, instance_id=item.instance_id, prompt_id=prompt_id,
                                       candidate_id=candidate_id, generation_seed=seed,
                                       source_index=item.source_index, source_instance_id=item.source_instance_id,
                                       max_new_tokens=budget, validation=result.to_dict(),
                                       input_encodes_precedence=item.input_encodes_precedence,
                                       configuration_sha256=manifest["configuration_sha256"])
                            _append_record(stream, row)
                            progress[key] = row
                        if time.monotonic() - last_summary_at >= 30:
                            atomic_json(output_dir / "summary.json", summarize(progress, items, config, manifest))
                            last_summary_at = time.monotonic()
                        print(json.dumps({"instance": item.source_index, "prompt": prompt_id, "candidate_slots_done": len(progress),
                                          "expected": manifest["expected_candidate_slots"]}), flush=True)
    except BaseException as exc:
        if engine is not None and hasattr(engine, "runtime_info"):
            atomic_json(output_dir / "runtime_device.json", dict(engine.runtime_info(), elapsed_seconds=time.monotonic() - started))
        atomic_json(output_dir / "summary.json", summarize(progress, items, config, manifest))
        with (output_dir / "failures.jsonl").open("a") as stream:
            _append_record(stream, {"utc": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
                                    "message": str(exc), "terminal_candidate_slots": len(progress),
                                    "is_cuda_oom": "out of memory" in str(exc).lower()})
        raise
    summary = summarize(progress, items, config, manifest)
    if hasattr(engine, "runtime_info"):
        atomic_json(output_dir / "runtime_device.json", dict(engine.runtime_info(), elapsed_seconds=time.monotonic() - started))
    atomic_json(output_dir / "summary.json", summary)
    if len(progress) != manifest["expected_candidate_slots"]:
        raise RuntimeError("Evaluation finished without all frozen candidate slots")
    atomic_json(output_dir / "COMPLETE", {"configuration_sha256": manifest["configuration_sha256"], "status": summary["status"], "candidate_slots": len(progress)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt-ids", default="seen_1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="PILOT ONLY: every artifact is marked PILOT_LIMITED")
    parser.add_argument("--input-field", default="prompt_machines_first")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    arguments = vars(parser.parse_args())
    arguments["prompt_ids"] = tuple(value.strip() for value in arguments["prompt_ids"].split(","))
    print(json.dumps(run(EvaluationConfig(**arguments)), indent=2), flush=True)


if __name__ == "__main__":
    main()
