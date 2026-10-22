#!/usr/bin/env python3
"""Controlled Phi-3 checkpoint experiment; no training occurs on import.

Input is unpadded JSONL containing ``input_ids`` and optionally all-one
``attention_mask``. All conversation tokens contribute to the causal LM loss,
except EOS/padding tokens masked by DataCollatorForLanguageModeling, matching
the recovered training script. Oversize examples are rejected, never truncated.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any


REQUESTED_TARGETS = ("qkv_proj", "o_proj", "fc1", "fc2")
SOURCE_DISCREPANCIES = {
    "local_alternate_script": {
        "source": "/mnt/volatile/users/habgaryan/litgpt/efficient_llm_jssp/train_phi_3.py",
        "lora_rank": 64, "lora_alpha": 128, "epochs": 2,
        "sequence_cap": 10000, "save_eval_steps": 50,
        "group_by_length": True, "load_best_model_at_end": True,
        "status": "Local alternate script; not the authoritative public main-run protocol."},
    "manuscript_defaults": {"lora_rank": 128, "lora_alpha": 256, "epochs": 1,
                            "sequence_cap": 40000, "save_eval_steps": 25},
    "public_release_script": {"source": "original_release/train_phi3_lora_jssp.py",
                        "published_url": "https://github.com/starjob42/datasetjsp/blob/24096f99b4961cc322dedfa74e3de2d811631080/train_phi3_lora_jssp.py",
                        "lora_rank": 128, "lora_alpha": 256, "epochs": 2,
                        "training_records": 118000, "validation_records": 2000,
                        "sequence_cap": 40000, "save_eval_steps": 50,
                        "group_by_length": True},
    "recovered_checkpoint_evidence": {"source": "original_release/recovered_checkpoint_provenance.json",
                             "global_step": 1750, "planned_steps": 1799, "epochs": 1,
                             "lora_rank": 128, "lora_alpha": 256,
                             "split_inference": "Approximately 115k training/5k validation; inferred from saved trainer metadata, not verified split identities.",
                             "historical_split_not_recovered": True},
    "interpretation": "Fresh controlled run; not an exact replay of the historical run.",
}


class PreflightError(ValueError):
    """An actionable configuration or environment failure before training."""


def parse_milestones(value: str, total_steps: int) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise PreflightError("Milestones must be comma-separated integer steps.") from exc
    if total_steps < 1:
        raise PreflightError("total_steps must be positive.")
    if steps != tuple(sorted(set(steps))) or not steps or steps[0] != 0:
        raise PreflightError("Milestones must be unique, increasing, and start at 0.")
    if steps[-1] != total_steps or any(step < 0 or step > total_steps for step in steps):
        raise PreflightError("Milestones must end at total_steps and lie within the run.")
    return steps


@dataclass(frozen=True)
class RunConfig:
    base_path: str
    tokenized_data: str
    output_dir: str
    validation_data: str | None = None
    total_steps: int = 1844
    milestones: tuple[int, ...] = (0, 461, 922, 1383, 1844)
    precision: str = "nf4-fp16"
    max_sequence_length: int = 40000
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    warmup_steps: int = 1
    seed: int = 42
    eval_steps: int = 25
    group_by_length: bool = True
    attention_implementation: str = "sdpa"
    pilot: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def protocol(self) -> str:
        protocol = ("phi3_manuscript_nf4_fp16_fresh_run" if self.precision == "nf4-fp16"
                    else "diagnostic_bf16_lora_not_manuscript_protocol")
        return "pilot_only:" + protocol if self.pilot else protocol

    def validate(self) -> None:
        parse_milestones(",".join(map(str, self.milestones)), self.total_steps)
        if self.precision not in {"nf4-fp16", "bf16-lora"}:
            raise PreflightError("Use nf4-fp16 or explicitly select diagnostic bf16-lora.")
        for key in ("max_sequence_length", "lora_rank", "lora_alpha", "micro_batch_size",
                    "gradient_accumulation_steps", "eval_steps"):
            if getattr(self, key) < 1:
                raise PreflightError(f"{key} must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise PreflightError("learning_rate must be positive and finite.")
        if not 0 <= self.lora_dropout < 1:
            raise PreflightError("lora_dropout must lie in [0, 1).")
        if self.warmup_steps < 0 or self.warmup_steps > self.total_steps:
            raise PreflightError("warmup_steps must lie between 0 and total_steps.")
        if self.effective_batch_size != 64:
            raise PreflightError("Controlled run requires effective batch size 64 on one GPU.")
        if self.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise PreflightError("Unsupported attention implementation.")
        if not Path(self.base_path).is_dir():
            raise PreflightError(f"Base model directory does not exist: {self.base_path}")


def read_record(raw: bytes, location: str, max_length: int) -> dict[str, list[int]]:
    try:
        row = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise PreflightError(f"Invalid JSON at {location}: {exc}") from exc
    if not isinstance(row, dict):
        raise PreflightError(f"Expected a JSON object at {location}.")
    if "labels" in row:
        raise PreflightError(f"Precomputed labels are not allowed at {location}; all-token loss is required.")
    ids = row.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise PreflightError(f"Nonempty input_ids list required at {location}.")
    if any(type(token) is not int or token < 0 for token in ids):
        raise PreflightError(f"input_ids must contain nonnegative integers at {location}.")
    if len(ids) < 2:
        raise PreflightError(f"At least two tokens are needed for causal loss at {location}.")
    if len(ids) > max_length:
        raise PreflightError(f"Sequence at {location} has {len(ids)} tokens, exceeding {max_length}; no truncation is performed.")
    mask = row.get("attention_mask", [1] * len(ids))
    if not isinstance(mask, list) or len(mask) != len(ids) or any(type(x) is not int or x != 1 for x in mask):
        raise PreflightError(f"attention_mask must be all-one and match unpadded input_ids at {location}.")
    return {"input_ids": ids, "attention_mask": mask}


class TokenizedJSONL:
    """Validated, random-access JSONL without retaining billions of Python ints."""

    def __init__(self, path: str | Path, max_length: int):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise PreflightError(f"Tokenized data file does not exist: {self.path}")
        self.max_length = max_length
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.max_token_id = 0
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line_number += 1
                digest.update(raw)
                if not raw.strip():
                    raise PreflightError(f"Blank JSONL row at {self.path}:{line_number}.")
                row = read_record(raw, f"{self.path}:{line_number}", max_length)
                self.offsets.append(offset)
                self.lengths.append(len(row["input_ids"]))
                self.max_token_id = max(self.max_token_id, max(row["input_ids"]))
        if not self.offsets:
            raise PreflightError(f"Tokenized dataset is empty: {self.path}")
        self.sha256 = digest.hexdigest()
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return read_record(self._handle.readline(), f"{self.path}:record {index}", self.max_length)

    def summary(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "records": len(self),
                "tokens": sum(self.lengths), "min_length": min(self.lengths),
                "max_length": max(self.lengths), "max_token_id": self.max_token_id}


def resolve_targets(module_names: list[str]) -> dict[str, Any]:
    matched = {target: sorted(name for name in module_names if name.rsplit(".", 1)[-1] == target)
               for target in REQUESTED_TARGETS}
    required = {"qkv_proj", "o_proj"}
    if any(not matched[target] for target in required):
        raise PreflightError("Phi-3 attention targets qkv_proj and o_proj must both exist; refusing another architecture.")
    if bool(matched["fc1"]) != bool(matched["fc2"]):
        raise PreflightError("Only one of fc1/fc2 exists; refusing an ambiguous LoRA target configuration.")
    selected = [target for target in REQUESTED_TARGETS if matched[target]]
    return {"requested": list(REQUESTED_TARGETS), "selected": selected,
            "absent": [target for target in REQUESTED_TARGETS if not matched[target]],
            "matched_module_names": {target: matched[target] for target in selected}}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_safe_training_state(folder: Path, state: dict[str, Any]) -> None:
    """Save nested optimizer/RNG state without a pickle deserializer on resume."""
    import numpy as np
    import torch
    from safetensors.torch import save_file
    tensors = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(tensors)}"
            tensors[name] = value.detach().cpu().contiguous().clone()
            return {"kind": "tensor", "name": name}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                raise TypeError("Object/structured arrays are unsupported in safe training state.")
            return {"kind": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape),
                    "data": base64.b64encode(value.tobytes()).decode("ascii")}
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [encode(item) for item in value]}
        if value is None or type(value) in (str, int, bool):
            return {"kind": "scalar", "value": value}
        if type(value) is float and math.isfinite(value):
            return {"kind": "scalar", "value": value}
        raise TypeError(f"Unsupported safe training state value: {type(value).__name__}")

    tree = encode(state)
    target = folder / "training_state.safetensors"
    temporary = target.with_name(target.name + ".tmp")
    save_file(tensors, str(temporary))
    temporary.replace(target)
    atomic_json(folder / "training_state.json", {"schema_version": 1, "tree": tree})


def load_safe_training_state(folder: Path) -> dict[str, Any]:
    import numpy as np
    from safetensors.torch import load_file
    metadata_tree = json.loads((folder / "training_state.json").read_text())
    if metadata_tree.get("schema_version") != 1:
        raise PreflightError("Unknown safe training state format.")
    tensors = load_file(str(folder / "training_state.safetensors"), device="cpu")

    def decode(value):
        kind = value["kind"]
        if kind == "scalar":
            scalar = value["value"]
            if scalar is not None and type(scalar) not in (str, int, bool, float):
                raise PreflightError("Invalid scalar in safe training state.")
            return scalar
        if kind == "tensor":
            return tensors[value["name"]]
        if kind == "ndarray":
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or dtype.fields:
                raise PreflightError("Unsafe array dtype in training state.")
            raw = base64.b64decode(value["data"], validate=True)
            return np.frombuffer(raw, dtype=dtype).reshape(value["shape"]).copy()
        if kind in ("tuple", "list"):
            values = [decode(item) for item in value["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "dict":
            return {decode(k): decode(v) for k, v in value["items"]}
        raise PreflightError(f"Invalid node kind in safe training state: {kind}")

    return decode(metadata_tree["tree"])


def examples_at_step(step: int, records: int, micro_batch: int = 2,
                     accumulation: int = 32) -> int:
    """Count sampled examples, including an incomplete accumulation at epoch end."""
    updates_per_epoch = math.ceil(math.ceil(records / micro_batch) / accumulation)
    epochs, within_epoch = divmod(step, updates_per_epoch)
    return epochs * records + min(within_epoch * micro_batch * accumulation, records)


def base_model_fingerprint(base: str) -> dict[str, Any]:
    folder = Path(base)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (folder / name).is_file():
            raise PreflightError(f"Base model is incomplete: missing {name}.")
    try:
        model_config = json.loads((folder / "config.json").read_text())
    except ValueError as exc:
        raise PreflightError(f"Invalid base config.json: {exc}") from exc
    if model_config.get("model_type") != "phi3":
        raise PreflightError("Base config.model_type must be phi3.")
    index = folder / "model.safetensors.index.json"
    if index.exists():
        try:
            expected_weights = set(json.loads(index.read_text())["weight_map"].values())
        except (ValueError, KeyError, AttributeError) as exc:
            raise PreflightError("Invalid model.safetensors.index.json.") from exc
    else:
        expected_weights = {"model.safetensors"}
    from safetensors import safe_open
    for name in expected_weights:
        path = folder / name
        if not path.is_file() or path.parent != folder:
            raise PreflightError(f"Base model is incomplete: missing weight shard {name}.")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                if not weights.keys():
                    raise ValueError("empty tensor file")
        except Exception as exc:
            raise PreflightError(f"Base weight shard {name} is incomplete or invalid: {exc}") from exc
    result = {}
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = folder / name
        if path.is_file():
            result[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size}
    result["weight_file_sizes"] = {str(path.relative_to(folder)): path.stat().st_size
                                    for path in sorted(folder.glob("*.safetensors"))}
    result["weight_file_sha256"] = {}
    for name in sorted(expected_weights):
        digest = hashlib.sha256()
        with (folder / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["weight_file_sha256"][name] = digest.hexdigest()
    return result


def dependency_preflight(precision: str) -> dict[str, str]:
    required = ["torch", "transformers", "peft", "accelerate"]
    if precision == "nf4-fp16":
        required.append("bitsandbytes")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise PreflightError("Missing training dependencies: " + ", ".join(missing)
                             + ". NF4 training never falls back to another precision.")
    return {name: metadata.version(name) for name in required}


def check_output_directory(output: Path, resume: str | None, resume_format: str = "safe") -> Path | None:
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"Output directory is nonempty: {output}; use an explicit resume or a fresh directory.")
        return None
    if not (output / "run_manifest.json").is_file():
        raise PreflightError("Resume requires the original output directory and run_manifest.json.")
    if (output / "training_complete.json").exists() or (output / "pilot_complete.json").exists():
        raise PreflightError("This experiment is already complete; use a fresh output directory for another run.")
    required_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    if resume_format == "safe":
        required_files += ["training_state.json", "training_state.safetensors", "experiment_checkpoint.json"]
    if resume == "latest":
        checkpoints = []
        for path in output.glob("checkpoint-*"):
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if path.is_dir() and all((path / item).is_file() for item in required_files):
                checkpoints.append((step, path))
        if not checkpoints:
            raise PreflightError("No complete resumable checkpoint was found.")
        return max(checkpoints)[1]
    checkpoint = Path(resume).resolve()
    if checkpoint.parent != output.resolve() or not checkpoint.is_dir():
        raise PreflightError("Resume checkpoint must be a checkpoint directory inside output-dir.")
    for name in required_files:
        if not (checkpoint / name).is_file():
            raise PreflightError(f"Resume checkpoint is incomplete: missing {name}.")
    try:
        selected_step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise PreflightError("Resume directory must be named checkpoint-STEP.") from exc
    for candidate in output.glob("checkpoint-*"):
        try:
            future_step = int(candidate.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if future_step > selected_step and (candidate / "experiment_checkpoint.json").is_file():
            raise PreflightError("A later complete checkpoint exists; resume latest to preserve all milestones.")
    return checkpoint


def assert_resume_compatible(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("config", "train_data", "validation_data", "base_model_files"):
        if previous.get(key) != current.get(key):
            raise PreflightError(f"Resume changes {key}; refusing to alter a controlled training trajectory.")


def train(config: RunConfig, resume: str | None = None, preflight_only: bool = False,
          resume_format: str = "safe", stop_after_step: int | None = None) -> dict[str, Any]:
    config.validate()
    if stop_after_step is not None:
        if not config.pilot or stop_after_step not in config.milestones or not 0 < stop_after_step < config.total_steps:
            raise PreflightError("Intentional --stop-after-step is restricted to an intermediate milestone in an explicit --pilot.")
    output = Path(config.output_dir).resolve()
    checkpoint = check_output_directory(output, resume, resume_format)
    versions = dependency_preflight(config.precision)
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("A CUDA GPU is required; no CPU training or precision fallback is performed.")
    if torch.cuda.device_count() != 1:
        raise PreflightError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES to keep effective batch size 64.")
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024 ** 3:
        raise PreflightError("Training requires a GPU with at least 40 GiB; check CUDA_DEVICE_ORDER/visible GPU UUID.")
    if checkpoint and resume_format == "native":
        from packaging.version import Version
        if Version(torch.__version__.split("+")[0]) < Version("2.6"):
            raise PreflightError("Native pickle resume requires torch>=2.6. Use the default safe companion state.")
    # One process/GPU is intentional: accumulation alone defines effective batch64.
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise PreflightError("This controlled run supports one GPU process only.")
    if config.precision == "bf16-lora" and not torch.cuda.is_bf16_supported():
        raise PreflightError("Explicit BF16 diagnostic requested, but this GPU lacks BF16 support.")
    torch.cuda.reset_peak_memory_stats()
    dataset = TokenizedJSONL(config.tokenized_data, config.max_sequence_length)
    validation = (TokenizedJSONL(config.validation_data, config.max_sequence_length)
                  if config.validation_data else None)
    if validation and dataset.sha256 == validation.sha256:
        raise PreflightError("Training and validation files are identical.")
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "preflight_passed", "protocol": config.protocol,
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": json.loads(json.dumps(asdict(config))), "versions": versions,
        "source_discrepancies": SOURCE_DISCREPANCIES,
        "base_model_files": base_model_fingerprint(config.base_path),
        "train_data": dataset.summary(), "validation_data": validation.summary() if validation else None,
        "effective_batch_size": config.effective_batch_size,
        "nominal_examples_at_final_step": config.total_steps * config.effective_batch_size,
        "actual_examples_at_final_step": examples_at_step(config.total_steps, len(dataset),
                                                          config.micro_batch_size, config.gradient_accumulation_steps),
        "loss": "All conversation tokens; EOS/pad IDs masked by DataCollatorForLanguageModeling(mlm=False).",
        "device": {"name": torch.cuda.get_device_name(0),
                   "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "milestones": {},
    }
    if checkpoint:
        previous = json.loads((output / "run_manifest.json").read_text())
        assert_resume_compatible(previous, manifest)
        manifest["milestones"] = previous.get("milestones", {})
        manifest["training_logs"] = previous.get("training_logs", [])
        manifest["resume_history"] = previous.get("resume_history", []) + [str(checkpoint)]
        manifest["metrics_scope"] = "Final Trainer metrics cover this resumed process; use saved per-step logs for the full trajectory."
    if preflight_only:
        return manifest

    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForLanguageModeling, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(config.seed)
    random.seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.base_path, trust_remote_code=False,
                                              local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise PreflightError("Base tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = config.max_sequence_length
    for split in (dataset, validation):
        if split and split.max_token_id >= len(tokenizer):
            raise PreflightError(f"Dataset token ID {split.max_token_id} exceeds tokenizer vocabulary {len(tokenizer)}.")
    load_args: dict[str, Any] = {
        "trust_remote_code": False, "local_files_only": True,
        "attn_implementation": config.attention_implementation,
        "torch_dtype": torch.float16 if config.precision == "nf4-fp16" else torch.bfloat16,
        "device_map": {"": 0},
    }
    if config.precision == "nf4-fp16":
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False)
    model = AutoModelForCausalLM.from_pretrained(config.base_path, **load_args)
    if getattr(model.config, "model_type", None) != "phi3":
        raise PreflightError("Base model config.model_type must be phi3.")
    targets = resolve_targets([name for name, _ in model.named_modules()])
    if config.precision == "nf4-fp16":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=targets["selected"], bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    manifest["lora_targets"] = targets
    manifest["model_implementation"] = {"class": type(model.base_model.model).__name__,
                                        "trust_remote_code": False, "library": "transformers_native"}
    manifest["parameters"] = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_tensor_elements": sum(p.numel() for p in model.parameters()),
        "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
    }
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    manifest["status"] = "training"
    manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "run_manifest.json", manifest)
    args = TrainingArguments(
        output_dir=str(output), max_steps=config.total_steps,
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear", optim="paged_adamw_8bit" if config.precision == "nf4-fp16" else "adamw_torch",
        fp16=config.precision == "nf4-fp16", bf16=config.precision == "bf16-lora",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": True},
        eval_strategy="steps" if validation else "no", eval_steps=config.eval_steps,
        logging_steps=config.eval_steps, save_strategy="no", save_total_limit=None,
        save_only_model=False, load_best_model_at_end=False, group_by_length=config.group_by_length,
        seed=config.seed, data_seed=config.seed, report_to="none", dataloader_num_workers=0,
        remove_unused_columns=True, overwrite_output_dir=False,
    )

    class ControlledTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            train_dataset = self.train_dataset if train_dataset is None else train_dataset
            if self.args.group_by_length and isinstance(train_dataset, TokenizedJSONL):
                from transformers.trainer_pt_utils import LengthGroupedSampler
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    lengths=train_dataset.lengths)
            return super()._get_train_sampler(train_dataset)

        def _safe_state(self, checkpoint_folder):
            if getattr(self, "_cached_safe_state_path", None) != checkpoint_folder:
                self._cached_safe_state = load_safe_training_state(Path(checkpoint_folder))
                self._cached_safe_state_path = checkpoint_folder
            return self._cached_safe_state

        def _load_optimizer_and_scheduler(self, checkpoint_folder):
            if checkpoint_folder and resume_format == "safe":
                state = self._safe_state(checkpoint_folder)
                self.optimizer.load_state_dict(state["optimizer"])
                self.lr_scheduler.load_state_dict(state["scheduler"])
                return
            return super()._load_optimizer_and_scheduler(checkpoint_folder)

        def _load_scaler(self, checkpoint_folder):
            if checkpoint_folder and resume_format == "safe":
                scaler_state = self._safe_state(checkpoint_folder)["scaler"]
                scaler = getattr(self.accelerator, "scaler", None)
                if scaler_state is not None:
                    if scaler is None:
                        raise PreflightError("Checkpoint has an AMP scaler but this run does not.")
                    scaler.load_state_dict(scaler_state)
                return
            return super()._load_scaler(checkpoint_folder)

        def _load_rng_state(self, checkpoint_folder):
            if checkpoint_folder and resume_format == "safe":
                import numpy as np
                rng = self._safe_state(checkpoint_folder)["rng"]
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch_cpu"])
                torch.cuda.random.set_rng_state_all(rng["torch_cuda"])
                return
            return super()._load_rng_state(checkpoint_folder)

    class MilestoneCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.started = time.monotonic()
            self.start_step = 0
            self.previous_seconds = 0
            self.microbatches = 0

        def progress(self, step):
            self.microbatches += 1
            atomic_json(output / "training_heartbeat.json", {
                "global_step": step, "session_microbatches_finished": self.microbatches,
                "session_seconds": time.monotonic() - self.started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "utc": datetime.now(timezone.utc).isoformat(), "protocol": config.protocol})

        def record(self, step: int) -> None:
            import numpy as np
            folder = output / f"checkpoint-{step}"
            scaler = getattr(self.trainer.accelerator, "scaler", None)
            save_safe_training_state(folder, {
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch_cpu": torch.random.get_rng_state(),
                        "torch_cuda": torch.cuda.random.get_rng_state_all()},
            })
            entry = {"global_step": step, "epoch": self.trainer.state.epoch,
                     "nominal_examples": step * config.effective_batch_size,
                     "actual_examples": examples_at_step(step, len(dataset), config.micro_batch_size,
                                                          config.gradient_accumulation_steps),
                     "session_training_seconds": time.monotonic() - self.started,
                     "cumulative_training_seconds": self.previous_seconds + time.monotonic() - self.started,
                     "checkpoint": str(folder), "utc": datetime.now(timezone.utc).isoformat(),
                     "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "device": manifest["device"], "protocol": config.protocol}
            atomic_json(folder / "experiment_checkpoint.json", entry)
            manifest["milestones"][str(step)] = entry
            atomic_json(output / "run_manifest.json", manifest)

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.start_step = state.global_step
            self.previous_seconds = manifest["milestones"].get(str(state.global_step), {}).get(
                "cumulative_training_seconds", 0)
            if state.global_step == 0 and not (output / "checkpoint-0").exists():
                # Trainer initializes optimizer/scheduler/RNG before this callback.
                self.trainer._save_checkpoint(self.trainer.model, trial=None)
                self.record(0)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.progress(state.global_step)
            if state.global_step in config.milestones:
                control.should_save = True
            if stop_after_step is not None and state.global_step >= stop_after_step:
                control.should_training_stop = True
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            self.progress(state.global_step)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for name in ("loss", "grad_norm", "eval_loss", "train_loss"):
                    if name in logs and not math.isfinite(float(logs[name])):
                        raise RuntimeError(f"Nonfinite {name} at optimizer step {state.global_step}.")
                manifest.setdefault("training_logs", []).append({"global_step": state.global_step, **logs})
                atomic_json(output / "run_manifest.json", manifest)
            return control

        def on_save(self, args, state, control, **kwargs):
            self.record(state.global_step)
            return control

    callback = MilestoneCallback()
    trainer = ControlledTrainer(
        model=model, args=args, train_dataset=dataset, eval_dataset=validation,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer, callbacks=[callback])
    callback.trainer = trainer
    atomic_json(output / "training_arguments.json", trainer.args.to_dict())
    try:
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if stop_after_step is not None and trainer.state.global_step == stop_after_step:
            manifest["status"] = "pilot_paused_for_resume_test"
            manifest["last_global_step"] = trainer.state.global_step
            manifest["metrics"] = result.metrics
            atomic_json(output / "run_manifest.json", manifest)
            return manifest
        if trainer.state.global_step != config.total_steps:
            raise RuntimeError(f"Training stopped at {trainer.state.global_step}, expected {config.total_steps}.")
        if not math.isfinite(result.metrics.get("train_loss", float("nan"))):
            raise RuntimeError("Training did not produce a finite train_loss.")
        for step in config.milestones:
            folder = output / f"checkpoint-{step}"
            for name in ("adapter_config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                         "training_state.json", "training_state.safetensors", "experiment_checkpoint.json"):
                if not (folder / name).is_file():
                    raise RuntimeError(f"Milestone {step} is not fully resumable: missing {name}.")
        trainer.save_state()
        manifest["status"] = "pilot_complete" if config.pilot else "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["metrics"] = result.metrics
        atomic_json(output / "run_manifest.json", manifest)
        atomic_json(output / ("pilot_complete.json" if config.pilot else "training_complete.json"), {
            "global_step": trainer.state.global_step, "protocol": config.protocol,
            "final_checkpoint": str(output / f"checkpoint-{config.total_steps}"),
            "finished_utc": manifest["finished_utc"], "metrics": result.metrics})
        return manifest
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["last_global_step"] = trainer.state.global_step
        atomic_json(output / "run_manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--tokenized-data", required=True)
    parser.add_argument("--validation-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=1844)
    parser.add_argument("--milestones", default="0,461,922,1383,1844")
    parser.add_argument("--precision", choices=("nf4-fp16", "bf16-lora"), default="nf4-fp16",
                        help="bf16-lora is an explicit diagnostic using a different protocol/optimizer.")
    parser.add_argument("--max-sequence-length", type=int, default=40000)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--no-group-by-length", action="store_true")
    parser.add_argument("--attention-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--resume-from-checkpoint", help="Explicit checkpoint path, or latest complete checkpoint.")
    parser.add_argument("--resume-format", choices=("safe", "native"), default="safe",
                        help="Default restores safetensors/JSON companions; native resume requires torch>=2.6.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate data/dependencies/GPU without loading or training the model.")
    parser.add_argument("--pilot", action="store_true", help="Mark the run as an excluded pilot; write pilot_complete.json instead of a study completion sentinel.")
    parser.add_argument("--stop-after-step", type=int, help="For --pilot only: stop at an intermediate saved milestone to test resume under the same full schedule.")
    args = parser.parse_args(argv)
    try:
        data = Path(args.tokenized_data).resolve()
        validation = Path(args.validation_data).resolve() if args.validation_data else None
        if data.is_dir():
            validation = validation or (data / "validation.jsonl")
            data = data / "train.jsonl"
        config = RunConfig(
            base_path=str(Path(args.base_path).resolve()), tokenized_data=str(data),
            validation_data=str(validation) if validation else None,
            output_dir=str(Path(args.output_dir).resolve()), total_steps=args.total_steps,
            milestones=parse_milestones(args.milestones, args.total_steps), precision=args.precision,
            max_sequence_length=args.max_sequence_length, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            micro_batch_size=args.micro_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, seed=args.seed,
            eval_steps=args.eval_steps, group_by_length=not args.no_group_by_length,
            attention_implementation=args.attention_implementation, pilot=args.pilot)
        result = train(config, args.resume_from_checkpoint, args.preflight_only, args.resume_format, args.stop_after_step)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except PreflightError as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
