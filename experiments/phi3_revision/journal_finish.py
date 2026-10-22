"""Finish the journal derivative after the existing scientific publisher exits.

Default invocation only reads readiness. --execute waits, calls the existing
guarded journal bridge and final builder, and records awaiting_final_root_review.
It never starts training/inference, reruns the frozen publisher, submits a paper,
or treats automated packaging as human scientific approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
try:
    from . import journal_build as builder
    from . import journal_publication as bridge
except ImportError:
    import journal_build as builder
    import journal_publication as bridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERATED_AUTHOR_FILES = {"journal_authors.tex", "journal_declarations.tex",
                          "journal_title_acknowledgments.tex"}
BRIDGE_TARGETS = (*bridge.TARGETS, "rebuttal.tex")
CODE_FILES = ("journal_finish.py", "journal_publication.py", "journal_tables.py",
              "journal_build.py", "si_package.py", "si_readiness.py", "SI_REPRODUCTION.md")


class FinishError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return bridge.file_hash(path)


def hashed(value, key="receipt_sha256"):
    value = dict(value)
    value[key] = bridge.json_hash(value)
    return value


def read_json(path, *, self_key=None, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise FinishError(f"Unreadable required artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinishError("Expected JSON object: " + str(path))
    if self_key and value.get(self_key) != bridge.json_hash({k: v for k, v in value.items() if k != self_key}):
        raise FinishError("Invalid " + self_key + ": " + str(path))
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise FinishError(message)


@dataclass(frozen=True)
class Config:
    paper_dir: Path = ROOT / "LLMs_can_Schedule_paper"
    study_dir: Path = HERE / "study"
    runs_dir: Path = HERE / "runs"
    auto_protocol: Path = HERE / "auto_prompt/artifacts/PROTOCOL.json"
    publisher_receipt: Path = HERE / "auto_prompt/artifacts/publishing/COMPLETE.json"
    output_dir: Path = HERE / "artifacts/journal_finish"
    python: str = "/usr/bin/python3"
    compile_host: str = "ourasi"
    poll_seconds: float = 60

    def normalized(self):
        fields = asdict(self)
        for key in ("paper_dir", "study_dir", "runs_dir", "auto_protocol", "publisher_receipt", "output_dir"):
            fields[key] = Path(fields[key]).resolve()
        require(0 < self.poll_seconds <= 60, "Poll interval must be positive and at most60 seconds")
        require(self.compile_host in {"ourasi", "kaisertrot", "localhost"}, "Unknown native compile host")
        result = Config(**fields)
        for directory in (result.paper_dir, result.study_dir, result.runs_dir,
                          result.auto_protocol.parent, result.publisher_receipt.parent):
            require(not result.output_dir.is_relative_to(directory)
                    and not directory.is_relative_to(result.output_dir),
                    "Finisher evidence must be separate from paper/scientific/publisher directories")
        return result

    def serialized(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


@contextmanager
def execution_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "FINISH.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinishError("Another journal finisher or surviving builder child holds FINISH.lock") from exc
        # Deliberately close without LOCK_UN. A builder child inherits this open
        # description, keeping the lock after a parent crash until it also exits.
        yield stream.fileno()


def observe_publisher(config):
    """Require the publisher's final tail writes, not COMPLETE alone."""
    directory = config.publisher_receipt.parent
    published = read_json(config.publisher_receipt, self_key="receipt_sha256", optional=True)
    status = read_json(directory / "STATUS.json", optional=True)
    exit_record = read_json(directory / "EXIT.json", optional=True)
    if published:
        require(published.get("status") == "complete"
                and published.get("purpose") == "VERIFIED_AUTO_PROMPT_PUBLICATION"
                and published.get("protocol_sha256") == bridge.AUTO_PROTOCOL,
                "Publisher completion is a pilot, incomplete, or belongs to another protocol")
        evidence = published.get("evidence_dir")
        if status and status.get("status") == "failed" and status.get("evidence_dir") == evidence:
            raise FinishError("Frozen publisher failed during its completion tail; inspect matching FAILURE/EXIT records")
        if exit_record and exit_record.get("evidence_dir") == evidence and exit_record.get("exit_code") not in (None, 0):
            raise FinishError("Frozen publisher failed during its completion tail; successful publication is not settled")
        if not status or not exit_record or status.get("status") != "complete":
            return {"status": "waiting_publisher_tail", "publisher_complete_exists": True,
                    "publisher_status": status.get("status") if status else None}
        require(exit_record.get("exit_code") == 0, "Publisher reports a failed exit despite its completion receipt")
        require(status.get("evidence_dir") == exit_record.get("evidence_dir") == evidence,
                "Publisher COMPLETE, STATUS and EXIT refer to different attempts")
        immutable = Path(evidence) / "COMPLETE.json"
        require(immutable.is_file() and digest(immutable) == digest(config.publisher_receipt),
                "Publisher's immutable attempt receipt differs from its current receipt")
        return {"status": "publisher_settled", "publisher_receipt_file_sha256": digest(config.publisher_receipt),
                "evidence_dir": evidence}
    if status and status.get("status") == "failed":
        failure = read_json(Path(status["evidence_dir"]) / "FAILURE.json", optional=True) or {}
        raise FinishError("Frozen publisher failed at " + str(failure.get("stage", status.get("stage")))
                          + ": " + str(failure.get("error_message", "inspect publisher FAILURE/EXIT records")))
    if exit_record and exit_record.get("exit_code") not in (None, 0):
        raise FinishError("Frozen publisher exited unsuccessfully; inspect " + str(directory / "EXIT.json"))
    manager = read_json(config.runs_dir / "manager/status.json", optional=True)
    if manager and manager.get("status") in {"failed", "failed_incomplete", "interrupted_incomplete", "timeout_incomplete"}:
        raise FinishError("Primary manager reports " + manager["status"] + "; no scientific job is restarted by this finisher")
    return {"status": "waiting_frozen_publication", "publisher_complete_exists": False,
            "publisher_status": status.get("status") if status else None,
            "publisher_stage": status.get("stage") if status else None}


def verify_chain(config):
    return bridge.verify_publication_chain(config.paper_dir, config.study_dir, config.runs_dir,
                                           config.auto_protocol, config.publisher_receipt)


def readiness(config):
    config = config.normalized()
    observation = observe_publisher(config)
    if observation["status"] == "publisher_settled":
        verified = verify_chain(config)
        observation["full_publication_guards_checked"] = True
        observation["status"] = "ready_for_journal_build" if verified["prior"] else "ready_for_journal_bridge"
    else:
        observation["full_publication_guards_checked"] = False
    return dict(observation, execute=False, config=config.serialized(),
                training_or_inference_launched=False, final_root_review_complete=False)


def preserve_files(paths, directory):
    """Copy original bytes before any finisher-owned publication write."""
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, path in enumerate(paths):
        path = Path(path)
        if path.is_file():
            backup = directory / f"{index:02d}_{path.name}"
            shutil.copy2(path, backup)
            result[str(path.resolve())] = {"backup": str(backup), "sha256": digest(backup)}
    return result


def start_bridge_transaction(config, attempt, verified):
    require(not verified["prior"], "Existing valid bridge needs no new transaction")
    expected = bridge.render_fragments(verified["primary"], verified["automatic"],
        (config.paper_dir / "revision_results.tex").read_text(),
        (config.paper_dir / "auto_prompt_results.tex").read_text(), verified["precision"])
    expected["rebuttal.tex"] = bridge.transform_rebuttal((config.paper_dir / "rebuttal.tex").read_text(), verified["precision"])
    require(set(expected) == set(BRIDGE_TARGETS), "Unexpected bridge target set")
    backups = preserve_files([config.paper_dir / name for name in BRIDGE_TARGETS], attempt / "bridge_before")
    require(len(backups) == len(BRIDGE_TARGETS), "Bridge target missing before transaction")
    value = {"purpose": "JOURNAL_TEXT_TRANSACTION", "paper_dir": str(config.paper_dir),
             "created_at_utc": now(), "publisher_receipt_file_sha256": digest(config.publisher_receipt),
             "before": backups,
             "expected_sha256": {str((config.paper_dir / name).resolve()): hashlib.sha256(text.encode()).hexdigest()
                                  for name, text in expected.items()}}
    for path, expected_hash in verified["bindings"].items():
        require(digest(path) == expected_hash, "Evidence changed before bridge transaction: " + path)
    transaction = attempt / "BRIDGE_TRANSACTION.json"
    write_json(transaction, hashed(value))
    return transaction


def recover_uncommitted_bridge(config, transaction):
    """Rollback only provably known partial writes; reject all other changes."""
    transaction = Path(transaction)
    value = read_json(transaction, self_key="receipt_sha256")
    require(value.get("paper_dir") == str(config.paper_dir)
            and value.get("publisher_receipt_file_sha256") == digest(config.publisher_receipt),
            "Bridge transaction no longer binds this paper/publisher")
    receipt_path = config.paper_dir / "build" / bridge.RECEIPT
    if receipt_path.exists():
        return {"status": "bridge_receipt_present_no_rollback"}
    expected_paths = {str((config.paper_dir / name).resolve()) for name in BRIDGE_TARGETS}
    require(set(value["before"]) == set(value["expected_sha256"]) == expected_paths,
            "Bridge transaction target set changed")
    restore = []
    # Check every target and backup before restoring even one file.
    for path, saved in value["before"].items():
        backup = Path(saved["backup"])
        require(backup.resolve().is_relative_to(transaction.parent.resolve())
                and backup.is_file() and digest(backup) == saved["sha256"],
                "Bridge recovery backup is missing or changed: " + str(backup))
        require(Path(path).is_file(), "Bridge target was externally removed; manual recovery required: " + path)
        actual = digest(path)
        require(actual in {saved["sha256"], value["expected_sha256"][path]},
                "Unexpected paper edit; no bridge target restored. Manual recovery required: " + path)
        if actual != saved["sha256"]:
            restore.append((Path(path), backup))
    discarded = preserve_files([path for path, _ in restore], transaction.parent / "bridge_uncommitted_preserved")
    for path, backup in restore:
        temporary = path.with_name(path.name + ".finisher_recovery.tmp")
        shutil.copy2(backup, temporary)
        temporary.replace(path)
    result = {"status": "known_uncommitted_bridge_writes_restored", "at_utc": now(),
              "restored": [str(path) for path, _ in restore], "preserved_partial_outputs": discarded}
    write_json(transaction.parent / "BRIDGE_RECOVERY.json", hashed(result))
    return result


def snapshot_build_inputs(config, attempt):
    fixed = [config.paper_dir / name for name in builder.SOURCES if name not in GENERATED_AUTHOR_FILES]
    fixed += [config.paper_dir / "rebuttal.tex", config.paper_dir / "build" / bridge.RECEIPT,
              config.publisher_receipt, builder.ART / "AUTHOR_DECLARATIONS.json", builder.ART / "confirmed_declarations.tex"]
    require(all(path.is_file() for path in fixed), "A required final-build input is missing")
    stable = {str(path.resolve()): digest(path) for path in fixed}
    staging = attempt / "predicted_author_files"
    staging.mkdir()
    builder.author_files(staging, final=True)
    require(all(digest(path) == expected for path, expected in stable.items()), "Build inputs changed during author-file prediction")
    value = {"purpose": "JOURNAL_FINAL_BUILD_INPUT_SNAPSHOT", "created_at_utc": now(),
             "stable_sha256": stable,
             "generated_author_sha256": {name: digest(staging / name) for name in GENERATED_AUTHOR_FILES}}
    output_names = ["JOURNAL_BUILD.json", "manuscript.pdf", "rebuttal.pdf", "ESM_1.zip",
                    "LLMs_can_Schedule_journal_sources.zip", "LLMs_can_Schedule_journal_sources_pending_results.zip",
                    *GENERATED_AUTHOR_FILES]
    value["original_outputs"] = preserve_files([config.paper_dir / name for name in output_names], attempt / "build_before")
    path = attempt / "BUILD_INPUTS.json"
    write_json(path, hashed(value))
    return path


def check_build_inputs(snapshot, paper, *, generated_required):
    value = read_json(snapshot, self_key="receipt_sha256")
    for path, expected in value["stable_sha256"].items():
        require(Path(path).is_file() and digest(path) == expected,
                "Source/author evidence changed during build; final receipt cannot be accepted: " + path)
    for name, expected in value["generated_author_sha256"].items():
        path = Path(paper) / name
        before = value["original_outputs"].get(str(path.resolve()), {}).get("sha256")
        allowed = {expected} if generated_required else {expected, before}
        require(path.is_file() and digest(path) in allowed,
                "Unexpected generated-author source edit; manual review required: " + str(path))


def build_command(config, attempt):
    require(config.paper_dir == builder.PAPER.resolve(), "Existing builder CLI targets only the canonical paper directory")
    return [config.python, "-B", "-u", "-m", "experiments.phi3_revision.journal_build", "--final",
            "--host", config.compile_host, "--build", str(attempt / "native_build")]


def run_builder(config, attempt, lock_fd, on_start):
    command = build_command(config, attempt)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    with (attempt / "builder.stdout.txt").open("wb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        on_start(child.pid, command)
        while True:
            try:
                return child.wait(timeout=config.poll_seconds)
            except subprocess.TimeoutExpired:
                on_start(child.pid, command)


def zip_hash(archive, name):
    digest_value = hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def checked_zip(path):
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    require(len(names) == len(set(names)) and archive.testzip() is None,
            "Archive has duplicate/corrupt members: " + str(path))
    require(all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names),
            "Unsafe archive member path: " + str(path))
    return archive


def verify_final_build(config, verified):
    paper = config.paper_dir
    path = paper / "JOURNAL_BUILD.json"
    receipt = read_json(path, self_key="receipt_sha256")
    require(receipt.get("status") == "compiled_and_checked" and receipt.get("experimental_results_pending") is False,
            "Current journal build is pending or incomplete")
    bridge_path = paper / "build" / bridge.RECEIPT
    require(receipt.get("upstream_journal_integration_receipt_sha256") == digest(bridge_path)
            and receipt.get("frozen_publisher_receipt_file_sha256") == digest(config.publisher_receipt),
            "Final journal build does not bind the bridge/publisher file hashes")
    require(set(receipt.get("source_sha256", {})) == set(builder.SOURCES), "Final build omits an editable manuscript source")
    outputs = {str(path): digest(path), str(bridge_path): digest(bridge_path)}
    for name, expected in receipt["source_sha256"].items():
        require(digest(paper / name) == expected, "Current final manuscript source changed: " + name)
        outputs[str(paper / name)] = expected
    require(receipt.get("rebuttal_source_sha256") == digest(paper / "rebuttal.tex"), "Final response source changed")
    outputs[str(paper / "rebuttal.tex")] = receipt["rebuttal_source_sha256"]
    require(receipt.get("author_metadata", {}).get("author_declarations_sha256") == digest(builder.ART / "AUTHOR_DECLARATIONS.json"),
            "Author metadata changed after the current journal build")
    stages = receipt.get("compiler", {}).get("stages", [])
    require(receipt.get("compiler", {}).get("status") == "compiled" and len(stages) == 7
            and all(stage.get("returncode") == 0 for stage in stages), "Final native compile lacks successful stages")
    build_dir = Path(receipt["build_dir"])
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    for name in ("manuscript", "rebuttal"):
        check = receipt["pdf_checks"][name]
        require(check.get("undefined_references_or_glyphs") is False and check.get("overfull_boxes") is False
                and type(check.get("pages")) is int and check["pages"] > 0, "Final PDF inspection flags are incomplete")
        for pdf in (paper / (name + ".pdf"), build_dir / (name + ".pdf")):
            require(digest(pdf) == check["sha256"], "Final PDF differs from inspected native build: " + str(pdf))
            outputs[str(pdf)] = check["sha256"]
        log = build_dir / (name + ".log")
        require(digest(log) == check["final_log_sha256"], "Final compile log changed: " + str(log))
        outputs[str(log)] = check["final_log_sha256"]
    source = paper / "LLMs_can_Schedule_journal_sources.zip"
    require(Path(receipt["source_package"]["path"]).resolve() == source
            and digest(source) == receipt["source_package"]["sha256"], "Final source ZIP missing or changed")
    with checked_zip(source) as archive:
        require(set(archive.namelist()) == set(builder.SOURCES) | {"manuscript.pdf", "manuscript.bbl", "README.txt"}
                and all("/" not in name for name in archive.namelist()), "Final typesetting ZIP is incomplete or not flat")
        for name, expected in receipt["source_sha256"].items():
            require(zip_hash(archive, name) == expected, "Source ZIP contains stale text: " + name)
        require(zip_hash(archive, "manuscript.pdf") == receipt["pdf_checks"]["manuscript"]["sha256"]
                and zip_hash(archive, "manuscript.bbl") == digest(build_dir / "manuscript.bbl"), "Source ZIP PDF/BBL changed")
    outputs[str(source)] = digest(source)
    supplement = paper / "ESM_1.zip"
    require(receipt["supplementary"].get("contains_final_results") is True
            and Path(receipt["supplementary"]["path"]).resolve() == supplement
            and digest(supplement) == receipt["supplementary"]["sha256"], "Final scientific SI missing or changed")
    with checked_zip(supplement) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        require(manifest.get("manifest_sha256") == bridge.json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
                == receipt["supplementary"]["archive_manifest_sha256"]
                and manifest.get("contains_final_results") is True
                and manifest.get("frozen_scientific_bytes_rewritten") is False,
                "Final SI manifest is invalid or pending")
        require(set(archive.namelist()) == set(manifest["file_sha256"]) | {"MANIFEST.json"}, "SI contains unbound or missing members")
        for name, expected in manifest["file_sha256"].items():
            require(zip_hash(archive, name) == expected, "SI member changed: " + name)
        # Every bound raw scientific artifact retained by the original bridge
        # must appear in final SI. Large model/library binaries remain excluded.
        protocol = read_json(config.auto_protocol, self_key="protocol_sha256")
        raw_roots = (config.runs_dir / "evaluations", config.runs_dir / "manager",
                     Path(protocol["output_dir"]), HERE / "baselines/artifacts/production",
                     config.publisher_receipt.parent)
        for original, expected in verified["bindings"].items():
            original = Path(original)
            if (any(original.is_relative_to(folder) for folder in raw_roots)
                    and (original.suffix in {".json", ".jsonl"} or original.name == "COMPLETE")):
                name = original.relative_to(ROOT).as_posix()
                require(manifest["file_sha256"].get(name) == expected, "Final SI omits/changes verified scientific record: " + name)
        for required in ("runs/main/training_complete.json", "runs/main/run_manifest.json", "runs/main/training_arguments.json",
                         "study/integrated_results.json", "study/selected_prompt.json"):
            name = "experiments/phi3_revision/" + required
            require(manifest["file_sha256"].get(name) == digest(HERE / required), "Final SI omits/changes full training/selection evidence: " + name)
    outputs[str(supplement)] = digest(supplement)
    return {"build_receipt_file_sha256": digest(path), "build_receipt_selfhash": receipt["receipt_sha256"],
            "outputs_file_sha256": outputs, "supplementary_manifest_sha256": manifest["manifest_sha256"],
            "author_contributions_supplied": receipt["author_metadata"].get("contribution_roles_supplied") is True,
            "final_root_review_complete": False}


def execute(config, *, sleep=time.sleep, build=run_builder):
    config = config.normalized()
    with execution_lock(config.output_dir) as lock_fd:
        code = {name: digest(HERE / name) for name in CODE_FILES}
        plan = hashed({"config": config.serialized(), "source_code_sha256": code}, "configuration_sha256")
        plan_path = config.output_dir / "PLAN.json"
        if plan_path.exists():
            require(read_json(plan_path) == plan, "Finisher code/configuration changed; preserve evidence and obtain a new reviewed launch directory")
        else:
            write_json(plan_path, plan)
        state_path = config.output_dir / "STATUS.json"
        previous = read_json(state_path, optional=True) or {}
        attempt = config.output_dir / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + str(os.getpid()))
        attempt.mkdir()
        state = {"status": "running", "stage": "wait_frozen_publication", "pid": os.getpid(),
                 "started_at_utc": now(), "attempt_dir": str(attempt),
                 "configuration_sha256": plan["configuration_sha256"], "final_root_review_complete": False}
        # Preserve recovery pointers even if this new process dies before it
        # reaches the recovery stage. Losing them would orphan a partial write.
        for key in ("bridge_transaction", "build_input_snapshot"):
            if previous.get(key):
                state[key] = previous[key]
        def update(**fields):
            state.update(fields, checked_at_utc=now())
            write_json(state_path, state)
            with (config.output_dir / "events.jsonl").open("a") as stream:
                json.dump({"at_utc": state["checked_at_utc"], "status": state["status"],
                           "stage": state["stage"], "attempt_dir": str(attempt),
                           "upstream_status": state.get("upstream", {}).get("status"),
                           "builder_pid": state.get("builder_pid")}, stream, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        write_json(attempt / "LAUNCH.json", hashed(dict(state, config=config.serialized(), code_sha256=code)))
        # Successful attempt receipts remain immutable. A fresh failed retry
        # must not leave an old current completion marker looking authoritative.
        prior_complete = config.output_dir / "COMPLETE.json"
        if prior_complete.exists():
            shutil.copy2(prior_complete, attempt / "PRIOR_CURRENT_COMPLETE.json")
            prior_complete.unlink()
        update()
        try:
            while True:
                require(all(digest(HERE / name) == expected for name, expected in code.items()), "Finisher dependency source changed after launch")
                observation = observe_publisher(config)
                update(upstream=observation)
                if observation["status"] == "publisher_settled":
                    break
                sleep(config.poll_seconds)
            if previous.get("bridge_transaction") and not (config.paper_dir / "build" / bridge.RECEIPT).exists():
                update(stage="recover_known_uncommitted_bridge")
                recover_uncommitted_bridge(config, previous["bridge_transaction"])
            update(stage="verify_completed_publication_chain")
            verified = verify_chain(config)
            existing = config.paper_dir / "JOURNAL_BUILD.json"
            if verified["prior"] and existing.is_file() and read_json(existing).get("experimental_results_pending") is False:
                update(stage="verify_existing_final_build")
                if previous.get("build_input_snapshot"):
                    check_build_inputs(previous["build_input_snapshot"], config.paper_dir, generated_required=True)
                final = verify_final_build(config, verified)
            else:
                if not verified["prior"]:
                    transaction = start_bridge_transaction(config, attempt, verified)
                    update(stage="integrate_guarded_journal_results", bridge_transaction=str(transaction))
                    bridge.integrate_verified_results(config.paper_dir, config.study_dir, config.runs_dir,
                                                       config.auto_protocol, config.publisher_receipt)
                    verified = verify_chain(config)
                if previous.get("build_input_snapshot"):
                    check_build_inputs(previous["build_input_snapshot"], config.paper_dir, generated_required=False)
                snapshot = snapshot_build_inputs(config, attempt)
                update(stage="build_final_journal", build_input_snapshot=str(snapshot))
                def on_start(pid, command):
                    update(builder_pid=pid, builder_command=command)
                returncode = build(config, attempt, lock_fd, on_start)
                require(returncode == 0, f"Final journal builder exited{ returncode }; inspect {attempt / 'builder.stdout.txt'}")
                check_build_inputs(snapshot, config.paper_dir, generated_required=True)
                update(stage="verify_final_journal_packages")
                verified = verify_chain(config)
                final = verify_final_build(config, verified)
            require(all(digest(HERE / name) == expected for name, expected in code.items()), "Finisher dependency source changed during publication")
            result = hashed({"status": "awaiting_final_root_review", "purpose": "GUARDED_JOURNAL_DERIVATIVE_AUTOMATION",
                             "completed_at_utc": now(), "configuration_sha256": plan["configuration_sha256"],
                             "publisher_receipt_file_sha256": digest(config.publisher_receipt),
                             "bridge_receipt_file_sha256": digest(config.paper_dir / "build" / bridge.RECEIPT),
                             "final_build": final, "attempt_dir": str(attempt),
                             "scientific_approval_asserted": False, "submitted_or_published_to_journal": False,
                             "final_root_review_complete": False})
            write_json(attempt / "COMPLETE.json", result)
            write_json(config.output_dir / "COMPLETE.json", result)
            update(status="awaiting_final_root_review", stage="automated_journal_build_verified")
            exit_record = {"exit_code": 0, "status": "awaiting_final_root_review", "at_utc": now(), "attempt_dir": str(attempt)}
            write_json(attempt / "EXIT.json", exit_record)
            write_json(config.output_dir / "EXIT.json", exit_record)
            return result
        except BaseException as exc:
            recovery = None
            if state.get("bridge_transaction") and not (config.paper_dir / "build" / bridge.RECEIPT).exists():
                try:
                    recovery = recover_uncommitted_bridge(config, state["bridge_transaction"])
                except BaseException as restore_error:
                    recovery = {"status": "manual_recovery_required", "error": str(restore_error)}
            failure = {"status": "failed", "stage": state["stage"], "error_type": type(exc).__name__,
                       "error_message": str(exc), "at_utc": now(), "attempt_dir": str(attempt),
                       "bridge_recovery": recovery, "traceback": traceback.format_exc(),
                       "final_root_review_complete": False}
            write_json(attempt / "FAILURE.json", failure)
            update(status="failed", error_type=type(exc).__name__, error_message=str(exc), bridge_recovery=recovery)
            exit_record = {"exit_code": 1, "status": "failed", "stage": state["stage"], "at_utc": now(), "attempt_dir": str(attempt)}
            write_json(attempt / "EXIT.json", exit_record)
            write_json(config.output_dir / "EXIT.json", exit_record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("paper-dir", "study-dir", "runs-dir", "auto-protocol", "publisher-receipt", "output-dir"):
        parser.add_argument("--" + name, type=Path, default=getattr(Config(), name.replace("-", "_")))
    parser.add_argument("--python", default=Config().python)
    parser.add_argument("--compile-host", default="ourasi", choices=("ourasi", "kaisertrot", "localhost"))
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    config = Config(**{key: value for key, value in vars(args).items() if key != "execute"})
    try:
        value = execute(config) if args.execute else readiness(config)
        print(json.dumps(value, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc),
                          "final_root_review_complete": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
