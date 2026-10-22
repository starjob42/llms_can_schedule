#!/usr/bin/env python3
"""Durable study manager. Default is a dry-run; --execute launches production.

Remote jobs survive an SSH disconnect. Their logs, PIDs, exit codes, evaluation
completion sentinels, and manager state live in the shared workspace. No busy
GPU is reused, and prompt selection is frozen before final prompt testing.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

try:
    from .evaluate import atomic_json, json_hash, sha256_file
    from .core import PROMPTS, SYSTEM_PROMPT
except ImportError:
    from evaluate import atomic_json, json_hash, sha256_file
    from core import PROMPTS, SYSTEM_PROMPT


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PYTHON = "/home/lamsade/habgaryan/miniconda3/envs/unsloth_env/bin/python"
SEACOVE_UUID = "GPU-025e10e6-263f-d814-6dd5-added86fc8af"


@dataclass(frozen=True)
class Worker:
    name: str
    host: str
    gpu_index: int
    gpu_uuid: str | None = None


DEFAULT_WORKERS = (
    Worker("ourasi_0", "ourasi", 0, "GPU-64742cce-f22c-6ed7-e485-b573564d590e"),
    Worker("ourasi_1", "ourasi", 1, "GPU-ea206d5f-1bbf-1139-4968-0b0a6d90ea66"),
    Worker("kaisertrot_0", "kaisertrot", 0),
)


class TransientTransportError(RuntimeError):
    """Read-only SSH observation failed; preserve job state and retry later."""


@dataclass(frozen=True)
class ManagerConfig:
    study_dir: str = str(HERE / "study")
    runs_dir: str = str(HERE / "runs")
    base_path: str = str(HERE / "models" / "phi3-mini-128k")
    tokenized_data: str = str(HERE / "data" / "train.jsonl")
    validation_data: str = str(HERE / "data" / "validation.jsonl")
    python: str = DEFAULT_PYTHON
    repo_root: str = str(REPO)
    training_host: str = "seacove"
    training_gpu_uuid: str = SEACOVE_UUID
    recovered_adapter: str | None = None
    poll_seconds: float = 60
    timeout_days: float = 0
    paper_dir: str = str(REPO / "LLMs_can_Schedule_paper")
    tectonic: str = "/home/lamsade/habgaryan/phd_project/clean_graphopt/.tools/tectonic/tectonic"

    def validate(self) -> None:
        if not 0 < self.poll_seconds <= 60 or not math.isfinite(self.timeout_days) or self.timeout_days < 0:
            raise ValueError("Polling must be <=60 seconds; timeout_days is zero for unlimited or positive")
        if not self.training_gpu_uuid.startswith("GPU-"):
            raise ValueError("Training requires an explicit verified GPU UUID")


def read_study(directory: Path) -> dict:
    study = json.loads((directory / "STUDY.json").read_text())
    payload = {key: value for key, value in study.items() if key != "protocol_sha256"}
    if study.get("protocol_sha256") != json_hash(payload):
        raise ValueError("Frozen STUDY.json protocol hash mismatch")
    for name, digest in study["artifact_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Frozen study artifact changed: {name}")
    if study.get("prompt_registry_sha256") != json_hash({"prompts": [asdict(prompt) for prompt in PROMPTS], "system_prompt": SYSTEM_PROMPT}):
        raise ValueError("Current prompt registry disagrees with the frozen study")
    if study["checkpoint_steps"] != [0, 461, 922, 1383, 1844]:
        raise ValueError("Unexpected checkpoint schedule")
    if study["test_count"] != 216 or study["prompt_validation_count"] != 72:
        raise ValueError("Expected the frozen 216-test/72-validation study")
    return study


def eval_jobs(config: ManagerConfig, study: dict) -> list[dict]:
    jobs = [{"name": f"checkpoint_{step}", "step": step, "kind": "checkpoint_test",
             "data": "test.jsonl", "prompt_ids": [study["training_budget_comparison_prompt"]]}
            for step in study["checkpoint_steps"][:-1]]
    jobs.extend([
        {"name": "final_prompt_validation", "step": study["checkpoint_steps"][-1],
         "kind": "prompt_validation", "data": "prompt_validation.jsonl", "prompt_ids": study["prompt_ids"]},
        {"name": "final_prompt_test", "step": study["checkpoint_steps"][-1],
         "kind": "prompt_test", "data": "test.jsonl", "prompt_ids": study["prompt_ids"], "needs_prompt_selection": True},
    ])
    if config.recovered_adapter:
        jobs.append({"name": "recovered_1750", "step": None, "kind": "recovered_context",
                     "data": "test.jsonl", "prompt_ids": [study["training_budget_comparison_prompt"]]})
    return jobs


def training_command(config: ManagerConfig, study: dict) -> list[str]:
    recipe = study["training_recipe"]
    command = [config.python, "-m", "experiments.phi3_revision.train", "--base-path", config.base_path,
               "--tokenized-data", config.tokenized_data, "--validation-data", config.validation_data,
               "--output-dir", str(Path(config.runs_dir) / "main"),
               "--total-steps", str(recipe["optimizer_steps"]), "--milestones", ",".join(map(str, study["checkpoint_steps"])),
               "--precision", "nf4-fp16", "--micro-batch-size", str(recipe["micro_batch_size"]),
               "--gradient-accumulation-steps", str(recipe["gradient_accumulation_steps"]),
               "--lora-rank", str(recipe["lora_rank"]), "--lora-alpha", str(recipe["lora_alpha"]),
               "--lora-dropout", str(recipe["lora_dropout"]), "--learning-rate", str(recipe["learning_rate"]),
               "--warmup-steps", str(recipe["warmup_steps"]), "--max-sequence-length", str(recipe["sequence_cap"]),
               "--seed", str(recipe["seed"]), "--attention-implementation", "sdpa"]
    if (Path(config.runs_dir) / "main" / "run_manifest.json").exists():
        command.extend(["--resume-from-checkpoint", "latest", "--resume-format", "safe"])
    return command


def evaluation_command(config: ManagerConfig, study: dict, job: dict) -> list[str]:
    decoding = study["decoding"]
    validation = job["kind"] == "prompt_validation"
    adapter = config.recovered_adapter if job["step"] is None else str(Path(config.runs_dir) / "main" / f"checkpoint-{job['step']}")
    return [config.python, "-m", "experiments.phi3_revision.evaluate", "--base-path", config.base_path,
            "--adapter", adapter, "--data", str(Path(config.study_dir) / job["data"]),
            "--prompt-ids", ",".join(job["prompt_ids"]), "--output-dir", str(Path(config.runs_dir) / "evaluations" / job["name"]),
            "--candidate-count", str(decoding["candidate_count_prompt_validation" if validation else "candidate_count_test"]),
            "--batch-size", str(decoding["batch_size_prompt_validation" if validation else "batch_size_test"]),
            "--seed", str(decoding["seed"]), "--temperature", str(decoding["temperature"]),
            "--top-k", str(decoding["top_k"]), "--top-p", str(decoding["top_p"])]


def report_command(config: ManagerConfig, *, select_only: bool = False) -> list[str]:
    command = [config.python, str(HERE / "report.py"), "--study-dir", config.study_dir,
               "--runs-dir", config.runs_dir, "--paper-dir", config.paper_dir]
    return command + (["--select-prompt-only"] if select_only else [])


def checkpoint_ready(runs_dir: Path, step: int) -> bool:
    directory = runs_dir / "main" / f"checkpoint-{step}"
    marker = directory / "experiment_checkpoint.json"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text())
    if value.get("global_step") != step:
        raise ValueError(f"Checkpoint marker step mismatch: {marker}")
    return (directory / "adapter_config.json").is_file() and any((directory / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))


def evaluation_complete(directory: Path) -> bool:
    required = ["COMPLETE", "summary.json", "candidates.jsonl", "evaluation_manifest.json"]
    if not all((directory / name).is_file() for name in required):
        return False
    summary = json.loads((directory / "summary.json").read_text())
    marker = json.loads((directory / "COMPLETE").read_text())
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    if manifest.get("configuration_sha256") != json_hash({key: value for key, value in manifest.items() if key != "configuration_sha256"}):
        raise ValueError(f"Evaluation manifest checksum mismatch: {directory}")
    if summary.get("run_kind") != "FULL" or manifest.get("run_kind") != "FULL":
        raise ValueError(f"Pilot output cannot satisfy production completion: {directory}")
    hashes = {summary.get("configuration_sha256"), marker.get("configuration_sha256"), manifest.get("configuration_sha256")}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"Evaluation completion hash mismatch: {directory}")
    expected = manifest["expected_candidate_slots"]
    if summary.get("terminal_candidate_slots") != expected or marker.get("candidate_slots") != expected or summary.get("not_generated_slots") != 0:
        raise ValueError(f"Incomplete evaluation has a COMPLETE marker: {directory}")
    if summary.get("status") != "complete":
        raise ValueError(f"Evaluation contains context skips; resolve before paper integration: {directory}")
    return True


def parse_gpu_inventory(gpu_csv: str, process_csv: str) -> list[dict]:
    active_uuids = {row[0].strip() for row in csv.reader(io.StringIO(process_csv)) if row and row[0].strip().startswith("GPU-")}
    devices = []
    for row in csv.reader(io.StringIO(gpu_csv)):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi GPU inventory")
        index, uuid, name, total, used, utilization = [part.strip() for part in row]
        devices.append({"index": int(index), "uuid": uuid, "name": name,
                        "memory_total_mib": int(total), "memory_used_mib": int(used),
                        "utilization_percent": int(utilization), "has_compute_process": uuid in active_uuids})
    return devices


def idle_a6000(device: dict) -> bool:
    return ("A6000" in device["name"] and device["memory_total_mib"] >= 45000
            and not device["has_compute_process"] and device["memory_used_mib"] <= 256
            and device["utilization_percent"] <= 5)


class SSHTransport:
    def command(self, host: str, script: str, *, check: bool = True) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, script],
                                  text=True, capture_output=True, timeout=45, check=check)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise TransientTransportError(f"SSH observation/launch on {host} failed: {exc}") from exc

    def inventory(self, host: str) -> list[dict]:
        gpu = self.command(host, "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits")
        processes = self.command(host, "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits")
        return parse_gpu_inventory(gpu.stdout, processes.stdout)

    def alive(self, host: str, pid: int) -> bool:
        if type(pid) is not int or pid <= 0:
            raise ValueError("Invalid remote PID")
        result = self.command(host, "kill -0 " + str(pid), check=False)
        if result.returncode == 255:
            raise TransientTransportError(f"SSH connectivity failed while observing PID {pid} on {host}")
        if result.returncode not in (0, 1):
            raise RuntimeError(f"Unexpected remote process observation status {result.returncode} on {host}")
        return result.returncode == 0

    def launch(self, host: str, uuid: str, command: list[str], config: ManagerConfig, prefix: Path) -> int:
        env_command = ["env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=" + uuid,
                       "PYTHONPATH=" + str(HERE / "vendor"), *command]
        exit_file = prefix.with_suffix(".exit")
        wrapper = (shlex.join(env_command) + "; study_exit_code=$?; "
                   + "printf '%s\\n' \"$study_exit_code\" > " + shlex.quote(str(exit_file) + ".tmp")
                   + "; mv " + shlex.quote(str(exit_file) + ".tmp") + " " + shlex.quote(str(exit_file))
                   + "; exit \"$study_exit_code\"")
        script = ("cd " + shlex.quote(config.repo_root) + " || exit 1; nohup sh -c " + shlex.quote(wrapper)
                  + " > " + shlex.quote(str(prefix.with_suffix(".log"))) + " 2>&1 < /dev/null & "
                  + "study_job_pid=$!; printf '%s\\n' \"$study_job_pid\" > " + shlex.quote(str(prefix.with_suffix(".pid")))
                  + "; printf '%s\\n' \"$study_job_pid\"")
        result = self.command(host, script)
        return int(result.stdout.strip().splitlines()[-1])


def _training_complete(config: ManagerConfig) -> bool:
    path = Path(config.runs_dir) / "main" / "training_complete.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("pilot") or value.get("global_step") != 1844 or not checkpoint_ready(Path(config.runs_dir), 1844):
        raise ValueError("Training completion is a pilot or lacks the final complete checkpoint")
    manifest = json.loads((Path(config.runs_dir) / "main" / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("config", {}).get("pilot"):
        raise ValueError("Training manifest does not describe a complete production run")
    return True


def _prepared(config: ManagerConfig) -> bool:
    path = Path(config.tokenized_data).parent / "manifest.json"
    if not path.is_file() or not Path(config.tokenized_data).is_file() or not Path(config.validation_data).is_file():
        return False
    value = json.loads(path.read_text())
    return value.get("status") == "prepared" and value.get("training_count") == 118000 and value.get("validation_count") == 2000


def compile_manuscript(config: ManagerConfig) -> None:
    paper = Path(config.paper_dir)
    build = paper / "build"
    build.mkdir(exist_ok=True)
    environment = dict(os.environ, XDG_CACHE_HOME="/home/lamsade/habgaryan/.cache")
    result = subprocess.run([config.tectonic, "-C", "--keep-logs", "--keep-intermediates", "-o", str(build), "neurips_2024.tex"],
                            cwd=paper, text=True, capture_output=True, timeout=600, env=environment)
    (Path(config.runs_dir) / "manager" / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Manuscript compilation failed; see manager/compile.log")
    pdf = build / "neurips_2024.pdf"
    inspection_script = (
        "import json,sys; from pypdf import PdfReader; reader=PdfReader(sys.argv[1]); "
        "print(json.dumps({'pages':len(reader.pages),'text':'\\n'.join(page.extract_text() or '' for page in reader.pages[:2])}))"
    )
    inspection = subprocess.run(["/usr/bin/python3", "-c", inspection_script, str(pdf)],
                                text=True, capture_output=True, check=True, timeout=30)
    inspected = json.loads(inspection.stdout)
    extracted = inspected["text"]
    if inspected["pages"] <= 12 or "LLMs" not in extracted[:1500] or "Formatting Instructions" in extracted[:1500]:
        raise RuntimeError("Compiled PDF failed manuscript title/page/sample checks")
    target = paper / "neurips_2024.pdf"
    backup = paper / "neurips_2024.pre_revision.pdf"
    if target.exists() and not backup.exists():
        import shutil
        shutil.copy2(target, backup)
    import shutil
    temporary_target = target.with_name(target.name + ".tmp")
    shutil.copy2(pdf, temporary_target)
    temporary_target.replace(target)


class StudyManager:
    def __init__(self, config: ManagerConfig, *, transport=None, workers=DEFAULT_WORKERS):
        config.validate()
        self.config = config
        self.study = read_study(Path(config.study_dir))
        self.jobs = eval_jobs(config, self.study)
        self.transport = transport or SSHTransport()
        self.workers = tuple(workers)
        self.directory = Path(config.runs_dir) / "manager"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "status.json"
        stable = {"config": asdict(config), "workers": [asdict(worker) for worker in workers],
                  "study_protocol_sha256": self.study["protocol_sha256"], "jobs": self.jobs,
                  "code_sha256": {name: sha256_file(HERE / name) for name in ("core.py", "evaluate.py", "train.py", "run_study.py")}}
        stable["manager_configuration_sha256"] = json_hash(stable)
        plan_path = self.directory / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != stable:
            raise ValueError("Manager resume refused: frozen plan, code, or configuration changed")
        atomic_json(plan_path, stable)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"status": "prepared_not_launched", "started_unix": None, "worker_bindings": {},
                          "training": {"status": "pending"}, "jobs": {job["name"]: {"status": "pending"} for job in self.jobs},
                          "prompt_selection": "pending", "report": "pending", "pdf": "pending"}
            self.save()

    def save(self):
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.state_path, self.state)

    def event(self, action: str, **details):
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "action": action, **details}) + "\n")

    def _poll_job(self, state: dict, prefix: Path, complete: bool) -> None:
        if complete:
            state["status"] = "complete"
            return
        if state["status"] == "complete":
            raise RuntimeError(f"Previously completed job lost a required artifact: {prefix}")
        if state["status"] == "launching":
            if not prefix.with_suffix(".pid").is_file():
                raise RuntimeError(f"Interrupted launch has no durable PID: {prefix}; inspect before recovery")
            state["pid"] = int(prefix.with_suffix(".pid").read_text().strip())
            state["status"] = "running"
        if state["status"] != "running":
            return
        exit_file = prefix.with_suffix(".exit")
        if exit_file.is_file():
            code = int(exit_file.read_text().strip())
            state["exit_code"] = code
            state["status"] = "failed"
            raise RuntimeError(f"Job exited {code} without a valid completion marker: {prefix}")
        if not self.transport.alive(state["host"], state["pid"]):
            state["status"] = "failed"
            raise RuntimeError(f"Remote job disappeared without completion: {prefix}; inspect its log before explicit recovery")

    def _launch(self, state: dict, host: str, uuid: str, command: list[str], prefix: Path, worker_name: str):
        # Existing exit/PID evidence must be reviewed rather than overwritten.
        if prefix.with_suffix(".exit").exists() or prefix.with_suffix(".pid").exists():
            raise RuntimeError(f"Existing process evidence blocks accidental rerun: {prefix}")
        state.update(status="launching", host=host, gpu_uuid=uuid, worker=worker_name,
                     command=command, log=str(prefix.with_suffix(".log")))
        self.save()
        pid = self.transport.launch(host, uuid, command, self.config, prefix)
        state.update(status="running", host=host, gpu_uuid=uuid, worker=worker_name, pid=pid,
                     command=command, log=str(prefix.with_suffix(".log")))
        self.event("launched", host=host, gpu_uuid=uuid, pid=pid, command=command)
        self.save()

    def refresh_progress(self):
        heartbeat = Path(self.config.runs_dir) / "main" / "training_heartbeat.json"
        if heartbeat.is_file():
            self.state["training_progress"] = json.loads(heartbeat.read_text())
        for job in self.jobs:
            summary_path = Path(self.config.runs_dir) / "evaluations" / job["name"] / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                progress = {key: summary.get(key) for key in ("terminal_candidate_slots", "expected_candidate_slots", "status", "n_evaluated_instances")}
                progress["feasible_candidates_by_prompt"] = {prompt: result.get("feasible_candidates") for prompt, result in summary.get("by_prompt", {}).items()}
                self.state["jobs"][job["name"]]["progress"] = progress

    def tick(self) -> bool:
        """One non-sleeping scheduling pass; true only after final PDF succeeds."""
        config = self.config
        runs = Path(config.runs_dir)
        self.refresh_progress()
        self._poll_job(self.state["training"], self.directory / "training", _training_complete(config))
        for job in self.jobs:
            self._poll_job(self.state["jobs"][job["name"]], self.directory / job["name"], evaluation_complete(runs / "evaluations" / job["name"]))
        if self.state["training"]["status"] == "failed" or any(value["status"] == "failed" for value in self.state["jobs"].values()):
            raise RuntimeError("A previous job failed; inspect logs and recover explicitly")
        if self.state["training"]["status"] == "pending" and _prepared(config):
            devices = self.transport.inventory(config.training_host)
            device = next((device for device in devices if device["uuid"] == config.training_gpu_uuid), None)
            if device is None:
                raise ValueError("Configured seacove training GPU UUID was not found")
            if idle_a6000(device):
                self._launch(self.state["training"], config.training_host, device["uuid"], training_command(config, self.study), self.directory / "training", "training")
        if self.state["jobs"]["final_prompt_validation"]["status"] == "complete" and self.state["prompt_selection"] != "complete":
            command = report_command(config, select_only=True)
            result = subprocess.run(command, cwd=config.repo_root, text=True, capture_output=True, timeout=600)
            (self.directory / "prompt_selection.log").write_text(result.stdout + result.stderr)
            if result.returncode or not (Path(config.study_dir) / "selected_prompt.json").is_file():
                raise RuntimeError("Validation-only prompt selection failed; see manager/prompt_selection.log")
            self.state["prompt_selection"] = "complete"
            self.save()
        if self.state["prompt_selection"] == "complete" and not (Path(config.study_dir) / "selected_prompt.json").is_file():
            raise ValueError("Frozen validation-selected prompt artifact disappeared")
        busy = {value.get("worker") for value in self.state["jobs"].values() if value["status"] == "running"}
        for worker in self.workers:
            if worker.name in busy:
                continue
            ready = next((job for job in self.jobs if self.state["jobs"][job["name"]]["status"] == "pending"
                          and (job["step"] is None or checkpoint_ready(runs, job["step"]))
                          and (not job.get("needs_prompt_selection") or self.state["prompt_selection"] == "complete")), None)
            if ready is None:
                break
            devices = self.transport.inventory(worker.host)
            device = next((device for device in devices if device["index"] == worker.gpu_index), None)
            if device is None:
                raise ValueError(f"Worker GPU index missing: {worker.name}")
            expected_uuid = worker.gpu_uuid or self.state["worker_bindings"].get(worker.name)
            if expected_uuid and device["uuid"] != expected_uuid:
                raise ValueError(f"Worker GPU UUID changed: {worker.name}")
            if not idle_a6000(device):
                self.event("worker_busy_skip", worker=worker.name, device=device)
                continue
            self.state["worker_bindings"][worker.name] = device["uuid"]
            self._launch(self.state["jobs"][ready["name"]], worker.host, device["uuid"], evaluation_command(config, self.study, ready), self.directory / ready["name"], worker.name)
        all_complete = self.state["training"]["status"] == "complete" and all(value["status"] == "complete" for value in self.state["jobs"].values())
        if all_complete:
            if self.state["report"] != "complete":
                result = subprocess.run(report_command(config), cwd=config.repo_root, text=True, capture_output=True, timeout=600)
                (self.directory / "report.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError("Final guarded paper integration failed; see manager/report.log")
                self.state["report"] = "complete"
                self.save()
            if self.state["pdf"] != "complete":
                compile_manuscript(config)
                self.state["pdf"] = "complete"
            self.state["status"] = "complete"
            self.save()
            atomic_json(self.directory / "COMPLETE", {"study_protocol_sha256": self.study["protocol_sha256"], "status": "complete"})
            return True
        self.state["status"] = "running"
        self.save()
        return False

    def execute(self) -> None:
        import fcntl
        self._lock = (self.directory / "manager.lock").open("a")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another study manager is already running") from exc
        if self.state["started_unix"] is None:
            self.state["started_unix"] = time.time()
        self.state["status"] = "running"
        self.save()
        try:
            while self.config.timeout_days == 0 or time.time() - self.state["started_unix"] < self.config.timeout_days * 86400:
                try:
                    if self.tick():
                        return
                except TransientTransportError as exc:
                    self.state["last_transport_error"] = {"utc": datetime.now(timezone.utc).isoformat(), "message": str(exc)}
                    self.event("transient_transport_error_retry", message=str(exc))
                    self.save()
                time.sleep(self.config.poll_seconds)
            self.state["status"] = "timeout_incomplete"
            self.save()
            raise TimeoutError("Explicitly configured study deadline reached; jobs/logs/results preserved and paper integration withheld")
        except BaseException as exc:
            if self.state["status"] != "timeout_incomplete":
                self.state["status"] = "interrupted_incomplete" if isinstance(exc, KeyboardInterrupt) else "failed_incomplete"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            self.save()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(ManagerConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), default=default,
                            type=float if name in {"poll_seconds", "timeout_days"} else str)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize production job launch; default only writes/prints a concrete plan")
    arguments = vars(parser.parse_args())
    execute = arguments.pop("execute")
    manager = StudyManager(ManagerConfig(**arguments))
    if execute:
        manager.execute()
    else:
        print(json.dumps({"status": "prepared_not_launched", "training": training_command(manager.config, manager.study),
                          "evaluations": {job["name"]: evaluation_command(manager.config, manager.study, job) for job in manager.jobs},
                          "state": str(manager.state_path)}, indent=2))


if __name__ == "__main__":
    main()
