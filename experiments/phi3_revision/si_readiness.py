"""Read archived SI bytes and CPU command parsers; never execute experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HELP_MODULES = (
    "prepare", "freeze_study", "train", "evaluate", "run_study", "report",
    "auto_prompt.protocol", "auto_prompt.optimizer", "auto_prompt.evaluate",
    "auto_prompt.report", "auto_prompt.publish", "baselines.run",
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def verify_files(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest.get("manifest_sha256") != json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Archive manifest selfhash mismatch")
    for name, expected in manifest["file_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Archive member is missing or outside extraction: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Archive member hash mismatch: " + name)
    for binding in manifest["frozen_source_bindings"] + manifest["frozen_input_bindings"]:
        if manifest["file_sha256"].get(binding["archive_path"]) != binding["sha256"]:
            raise ValueError("Frozen source binding mismatch: " + binding["archive_path"])
    for binding in manifest["protocol_selfhashes"]:
        value = json.loads((root / binding["archive_path"]).read_text())
        if value.get(binding["key"]) != json_hash({k: v for k, v in value.items() if k != binding["key"]}):
            raise ValueError("Frozen protocol selfhash mismatch: " + binding["archive_path"])
    return manifest


def check_help(root):
    root = Path(root).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(root / "experiments/phi3_revision"))
    # -S excludes site-packages. An accidental Torch/Transformers dependency in
    # a help path therefore fails rather than importing a GPU-capable library.
    checks = []
    for module in HELP_MODULES:
        command = [sys.executable, "-S", "-B", "-m", "experiments.phi3_revision." + module, "--help"]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                text=True, timeout=30)
        if result.returncode or "usage:" not in result.stdout.lower():
            raise RuntimeError("CPU help failed for " + module + ": " + (result.stderr or result.stdout)[-1000:])
        checks.append({"module": module, "returncode": 0, "site_packages_disabled": True})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check-help", action="store_true")
    args = parser.parse_args()
    manifest = verify_files(args.root)
    value = {"status": "archive_bytes_verified", "scientific_completion_verified": False,
             "file_count": len(manifest["file_sha256"]),
             "manifest_sha256": manifest["manifest_sha256"],
             "contains_final_results": manifest["contains_final_results"]}
    if args.check_help:
        value["cpu_help_checks"] = check_help(args.root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
