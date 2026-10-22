"""Package a portable inspection tree without rewriting frozen science files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

PREFIX = "experiments/phi3_revision/"
SOURCE_FILES = (
    "core.py", "prepare.py", "freeze_study.py", "train.py", "evaluate.py",
    "report.py", "run_study.py", "README.md", "SI_REPRODUCTION.md",
    "si_package.py", "si_readiness.py",
    "test_core.py", "test_freeze_study.py", "test_train.py", "test_evaluate.py",
    "test_run_study.py", "test_report.py",
    "auto_prompt/__init__.py", "auto_prompt/protocol.py", "auto_prompt/optimizer.py",
    "auto_prompt/evaluate.py", "auto_prompt/report.py", "auto_prompt/publish.py",
    "auto_prompt/README.md", "auto_prompt/test_protocol.py", "auto_prompt/test_optimizer.py",
    "auto_prompt/test_evaluate.py", "auto_prompt/test_report.py",
    "baselines/__init__.py", "baselines/run.py", "baselines/report.py",
    "baselines/README.md", "baselines/test_baselines.py", "baselines/test_report.py",
    "original_release/README.md", "original_release/PROVENANCE.md",
    "original_release/requirements.txt", "original_release/infer_phi3.py",
    "original_release/train_phi3_lora_jssp.py", "original_release/audit_original_release.py",
    "original_release/utils/data_preprocessing.py", "original_release/utils/helping_functions.py",
    "original_release/utils/solution_generation.py",
)
ARTIFACT_FILES = (
    "study/STUDY.json", "study/test.jsonl", "study/prompt_validation.jsonl",
    "study/PROMPTS.json", "study/PREPARATION_SPLIT.json", "data/manifest.json",
    "auto_prompt/artifacts/PROTOCOL.json", "baselines/artifacts/PROTOCOL.json",
    "original_release/release_manifest.json", "original_release/recovered_checkpoint_provenance.json",
    "original_release/jssp_llm_format_120k.lfs_pointer.txt", "original_release/audit.json",
    "artifacts/base_revision.txt", "artifacts/production_training_preflight.json",
    "artifacts/PRECISION_CLARIFICATION.json",
)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_hash(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def final_record_paths(revision):
    """An explicit record-only allowlist: no weights, state binaries or caches."""
    paths = set()
    for folder in ("runs/evaluations", "runs/manager", "auto_prompt/artifacts/production",
                   "auto_prompt/artifacts/publishing", "baselines/artifacts/production"):
        for path in (revision / folder).rglob("*"):
            if path.is_file() and (path.suffix in {".json", ".jsonl"} or path.name == "COMPLETE"):
                paths.add(path)
    for name in ("runs/main/training_complete.json", "runs/main/run_manifest.json",
                 "runs/main/training_arguments.json", "study/integrated_results.json",
                 "study/selected_prompt.json"):
        path = revision / name
        if not path.is_file():
            raise ValueError("Final supporting record missing: " + str(path))
        paths.add(path)
    # Saved metadata are useful without including multi-GB adapter/state files.
    for path in (revision / "runs/main").glob("checkpoint-*/*.json"):
        paths.add(path)
    return sorted(paths)


def build_supplementary(paper, root, title, authors, metadata_path, *, final=False):
    paper, root = Path(paper), Path(root).resolve()
    revision = root / "experiments/phi3_revision"
    metadata = json.loads(Path(metadata_path).read_text())
    email = metadata["author_metadata"].get("corresponding_email")
    corresponding = [a for a in authors if a["email"] == email]
    if len(corresponding) != 1:
        raise ValueError("Supporting information needs one confirmed corresponding author")
    corresponding_name = corresponding[0]["given"] + " " + corresponding[0]["family"]
    members = {}
    for name in SOURCE_FILES + ARTIFACT_FILES:
        path = revision / name
        if not path.is_file():
            raise ValueError("Supporting source missing: " + str(path))
        members[PREFIX + name] = path.read_bytes()
    if final:
        for path in final_record_paths(revision):
            members[path.relative_to(root).as_posix()] = path.read_bytes()
    generated = {
        "experiments/__init__.py": b'"""Archive package initializer; no model imports."""\n',
        "experiments/phi3_revision/__init__.py": b'"""Archived revision package; scientific source files remain unchanged."""\n',
    }
    members.update(generated)
    study = json.loads(members[PREFIX + "study/STUDY.json"])
    auto = json.loads(members[PREFIX + "auto_prompt/artifacts/PROTOCOL.json"])
    baseline = json.loads(members[PREFIX + "baselines/artifacts/PROTOCOL.json"])
    release = json.loads(members[PREFIX + "original_release/release_manifest.json"])
    # Read the changing live manifest once. Only its recorded environment/model
    # facts are used in pending packaging, never a partial result population.
    run_path = revision / "runs/main/run_manifest.json"
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    preflight = json.loads(members[PREFIX + "artifacts/production_training_preflight.json"])
    if run["versions"] != preflight["versions"]:
        raise ValueError("Actual training versions differ from the archived preflight")
    baseline_env = baseline["provenance"]["dependencies"]
    environment = {
        "scope": "RECORDED_EXECUTION_ENVIRONMENTS_NOT_AN_INSTALL_LOCK",
        "main_run": {"source": PREFIX + "runs/main/run_manifest.json",
                     "source_archived_in_this_package": final,
                     "source_file_sha256_at_packaging": digest_bytes(run_bytes),
                     "matching_preflight_archive_path": PREFIX + "artifacts/production_training_preflight.json",
                     "versions": run["versions"], "device": run["device"],
                     "python_version_not_recorded_in_this_manifest": True,
                     "precision": run["config"]["precision"],
                     "model_implementation": run["model_implementation"]},
        "legacy_baselines": {"source": PREFIX + "baselines/artifacts/PROTOCOL.json",
                             "python": baseline_env["python"], "platform": baseline_env["platform"],
                             "versions": {name: value["version"] for name, value in baseline_env["packages"].items()},
                             "fingerprint_scope": baseline_env["fingerprint_scope"]},
        "limitations": ["No packages installed or imported to build this archive.",
                        "Main manifest records five core versions, not a complete environment or CUDA/driver lock.",
                        "Binary vendor dependencies and external legacy libraries must be supplied independently."],
    }
    provenance = {
        "scope": "PINNED_ORIGINAL_INPUT_IDENTITIES_NO_DOWNLOAD_OR_REPLAY_PERFORMED",
        "dataset": {"repository": release["repository"], "commit": release["commit"],
                    "files": release["files"],
                    "manifest_archive_path": PREFIX + "original_release/release_manifest.json"},
        "base_model": {"repository": auto["base_model"], "revision": auto["base_revision"],
                       "files": auto["base_artifacts"]["files"],
                       "artifact_sha256": auto["base_artifacts"]["sha256"]},
        "baselines": {"repository": "https://github.com/AndreaCorsini1/SelfLabelingJobShop",
                      "commit": baseline["provenance"]["git_head"],
                      "source_hashes": baseline["provenance"]["source_python_sha256"],
                      "weights": baseline["provenance"]["weights"],
                      "protocol_archive_path": PREFIX + "baselines/artifacts/PROTOCOL.json"},
        "original_protocols": {"primary": study["protocol_sha256"],
                               "automatic_refinement": auto["protocol_sha256"],
                               "baselines": baseline["protocol_sha256"]},
        "path_policy": "Original absolute-path JSON is preserved byte-exact. A new execution requires new paths, new freezes and new protocol hashes; this archive performs no such execution.",
        "excluded": ["full source/tokenized training data", "base and adapter weights",
                     "optimizer/RNG binaries", "external baseline libraries/weights",
                     "binary dependencies", "local compiler/cache"],
    }
    members["ENVIRONMENT.json"] = encoded(environment)
    members["PROVENANCE.json"] = encoded(provenance)
    members["REPRODUCTION.md"] = members[PREFIX + "SI_REPRODUCTION.md"]
    cover = (title + "\nAnnals of Mathematics and Artificial Intelligence\n" +
             "; ".join(a["given"] + " " + a["family"] for a in authors) + "\n" +
             "Corresponding author: " + corresponding_name + "\n" +
             "Corresponding affiliation: LAMSADE, CNRS, Université Paris Dauphine–PSL, Paris, France\n" +
             "Corresponding email: " + email + "\n\n" +
             ("COMPLETED-STUDY RECORDS. Final build barriers were checked by the journal builder before packaging. Archive byte checks alone do not independently revalidate results.\n" if final else
              "METHODS AND FROZEN INPUTS ONLY. The manuscript reports completed fixed-instruction checkpoint and baseline evaluations; final selected-instruction and automatic-refinement results remain in progress. This archive excludes the full raw record populations and does not assert final scientific completion.\n") +
             "\nOnline Resource1 caption: Frozen scheduling inputs, prompt and evaluation protocols, source snapshots, provenance and reproduction instructions" +
             (", with completed raw training/evaluation records.\n" if final else "; unfinished scientific output populations are excluded.\n") +
             "\nExtract the entire archive. From its root, first read REPRODUCTION.md, then run:\n" +
             "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/phi3_revision/si_readiness.py --check-help\n" +
             "This checks files and CPU help only, with no model, GPU, server, network or installation.\n" +
             "The original experiments/phi3_revision hierarchy is preserved; do not flatten it.\n" +
             "MANIFEST.json hashes every other member. ENVIRONMENT.json and PROVENANCE.json document actual recorded environments and pinned external origins.\n" +
             "Original absolute paths are immutable provenance. New-machine executions require newly frozen protocols, not editing archived JSON or reusing the original protocol hash.\n" +
             "Large training data, model/adapter weights, optimizer states and binary dependencies are excluded; retrieval identities are recorded.\n" +
             "The separate flat journal-source ZIP contains typesetting sources/PDF; this code ZIP is not the LaTeX source submission.\n")
    members["README.txt"] = cover.encode()
    frozen_bindings = []
    for original, expected in auto["source_sha256"].items():
        archive_path = PREFIX + original.split("/experiments/phi3_revision/", 1)[1]
        if archive_path not in members or digest_bytes(members[archive_path]) != expected:
            raise ValueError("Frozen automatic source changed or omitted: " + original)
        frozen_bindings.append({"archive_path": archive_path, "sha256": expected,
                                "bound_by": PREFIX + "auto_prompt/artifacts/PROTOCOL.json"})
    for archive_path, expected in baseline["provenance"]["harness_sha256"].items():
        if archive_path not in members or digest_bytes(members[archive_path]) != expected:
            raise ValueError("Frozen baseline source changed or omitted: " + archive_path)
        frozen_bindings.append({"archive_path": archive_path, "sha256": expected,
                                "bound_by": PREFIX + "baselines/artifacts/PROTOCOL.json"})
    protocol_hashes = []
    for name in ("study/STUDY.json", "auto_prompt/artifacts/PROTOCOL.json", "baselines/artifacts/PROTOCOL.json"):
        value = json.loads(members[PREFIX + name])
        if value.get("protocol_sha256") != json_hash({k: v for k, v in value.items() if k != "protocol_sha256"}):
            raise ValueError("Original protocol selfhash invalid: " + name)
        protocol_hashes.append({"archive_path": PREFIX + name, "key": "protocol_sha256"})
    input_bindings = []
    for name, expected in study["artifact_sha256"].items():
        archive_path = PREFIX + "study/" + name
        if archive_path not in members or digest_bytes(members[archive_path]) != expected:
            raise ValueError("Frozen primary input changed or omitted: " + archive_path)
        input_bindings.append({"archive_path": archive_path, "sha256": expected,
                               "bound_by": PREFIX + "study/STUDY.json"})
    manifest = {
        "schema_version": 1, "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SUPPLEMENTARY_ARCHIVE_BYTE_BINDINGS_NOT_SCIENTIFIC_REVALIDATION",
        "contains_final_results": final, "original_repository_layout_preserved": True,
        "frozen_scientific_bytes_rewritten": False,
        "file_sha256": {name: digest_bytes(data) for name, data in sorted(members.items())},
        "generated_documentation_members": sorted([*generated, "README.txt", "REPRODUCTION.md", "ENVIRONMENT.json", "PROVENANCE.json"]),
        "frozen_source_bindings": frozen_bindings, "frozen_input_bindings": input_bindings,
        "protocol_selfhashes": protocol_hashes,
    }
    manifest["manifest_sha256"] = json_hash(manifest)
    members["MANIFEST.json"] = encoded(manifest)
    target = paper / "ESM_1.zip"
    temporary = target.with_name(target.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(members.items()):
                archive.writestr(name, data)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target), "sha256": digest_bytes(target.read_bytes()),
            "contains_final_results": final, "archive_manifest_sha256": manifest["manifest_sha256"],
            "files": len(members), "original_repository_layout_preserved": True}
