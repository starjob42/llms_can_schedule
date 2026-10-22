#!/usr/bin/env python3
"""Read-only, fail-closed reporting for the separately frozen baseline study.

The legacy scheduling audit runs in its original Python environment. This
module can otherwise be imported by the Phi-3 reporting environment without
importing Torch, PyG, Gym, or any legacy policy implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.phi3_revision.core import (
    EvaluationRecord, ValidationResult, aggregate_results, canonical_hash,
    parse_matrix,
)

MODELS = ("l2d_20x20_1_99", "slj_b32", "slj_b128", "slj_b256")
MODEL_LABELS = {"l2d_20x20_1_99": "L2D (20$\\times$20, 1--99)",
                "slj_b32": "SLJ B32", "slj_b128": "SLJ B128", "slj_b256": "SLJ B256"}
PURPOSE = "SCIENTIFIC_FROZEN_216"


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


def _receipt_header(receipt, protocol):
    if receipt.get("status") != "complete" or receipt.get("purpose") != PURPOSE:
        raise ValueError("Baseline receipt is incomplete or a pilot")
    if (receipt.get("protocol_sha256") != protocol["protocol_sha256"]
            or receipt.get("instances") != 216 or receipt.get("candidates_per_instance") != 10
            or set(receipt.get("models", {})) != set(MODELS)):
        raise ValueError("Baseline receipt does not cover the frozen 216 x 10 x 4 population")
    for name in MODELS:
        item = receipt["models"][name]
        if item.get("candidates") != 2160 or len(item.get("raw_file_sha256", {})) != 216:
            raise ValueError(f"Baseline receipt lacks all candidates/raw files: {name}")


def _verify_completed_artifacts(study_dir, baseline_dir):
    study_dir, baseline_dir = Path(study_dir).resolve(), Path(baseline_dir).resolve()
    protocol_path = baseline_dir / "artifacts/PROTOCOL.json"
    output = baseline_dir / "artifacts/production"
    receipt_path = output / "COMPLETE.json"
    if not receipt_path.is_file():
        raise ValueError("Requested baseline production is incomplete: COMPLETE.json is absent")
    study = _self_hashed(study_dir / "STUDY.json", "protocol_sha256")
    protocol = _self_hashed(protocol_path, "protocol_sha256")
    receipt = _self_hashed(receipt_path, "receipt_sha256")
    if (Path(protocol["parent_study"]).resolve() != study_dir / "STUDY.json"
            or protocol["parent_protocol_sha256"] != study["protocol_sha256"]):
        raise ValueError("Baseline protocol belongs to a different parent study")
    if protocol.get("instances") != 216 or protocol.get("candidates_per_instance") != 10 or protocol.get("models") != list(MODELS):
        raise ValueError("Unexpected frozen baseline population or models")
    test = study_dir / "test.jsonl"
    if (Path(protocol["test_file"]).resolve() != test or _file_hash(test) != protocol["test_file_sha256"]
            or _file_hash(test) != study["artifact_sha256"]["test.jsonl"]):
        raise ValueError("Baseline test population changed")
    rows = [json.loads(line) for line in test.read_text().splitlines() if line.strip()]
    identities = [canonical_hash(parse_matrix(row["matrix"])) for row in rows]
    if (len(identities) != 216 or len(set(identities)) != 216 or identities != study["test_instance_ids"]
            or identities != protocol["instance_ids"] or identities != [row["instance_id"] for row in rows]):
        raise ValueError("Baseline canonical instance identities differ from the parent study")
    _receipt_header(receipt, protocol)
    run = json.loads((output / "RUN.json").read_text())
    if (run.get("purpose") != PURPOSE or run.get("protocol_sha256") != protocol["protocol_sha256"]
            or run.get("instance_ids") != identities or run.get("models") != list(MODELS)
            or run.get("candidates_per_instance") != 10 or run.get("device") != "cpu"):
        raise ValueError("Baseline production manifest differs from the frozen study")
    source = Path(protocol["provenance"]["source_root"])
    for relative, digest in protocol["provenance"]["source_python_sha256"].items():
        if _file_hash(source / relative) != digest:
            raise ValueError(f"Original baseline source changed: {relative}")
    for relative, digest in protocol["provenance"]["harness_sha256"].items():
        if _file_hash(PROJECT / relative) != digest:
            raise ValueError(f"Frozen baseline harness/core changed: {relative}")
    for name in MODELS:
        weight = protocol["provenance"]["weights"][name]
        if _file_hash(weight["path"]) != weight["sha256"]:
            raise ValueError(f"Baseline checkpoint changed: {name}")
        expected_files = {f"{name}/{identity}.jsonl" for identity in identities}
        recorded = receipt["models"][name]["raw_file_sha256"]
        if set(recorded) != expected_files:
            raise ValueError(f"Baseline raw-file population differs: {name}")
        for relative, digest in recorded.items():
            if _file_hash(output / relative) != digest:
                raise ValueError(f"Baseline raw evidence changed: {relative}")
    return study, protocol, receipt, identities


def _legacy_executable(protocol):
    # sys.executable.resolve() in the frozen fingerprint points to /usr/bin;
    # invoke the actual venv launcher so the legacy site-packages stay active.
    paths = protocol["provenance"]["dependencies"]["packages"]["torch"]["files"]
    for filename in paths:
        path = Path(filename)
        if path.name == "__init__.py" and path.parent.name == "torch":
            site = next((parent for parent in path.parents if parent.name == "site-packages"), None)
            if site is not None:
                candidate = site.parents[2] / "bin/python"
                if candidate.is_file():
                    return str(candidate)
    raise ValueError("Cannot identify the frozen legacy Python environment; supply legacy_python")


def _legacy_audit(study_dir, baseline_dir, protocol, legacy_python=None, legacy_host=None):
    python = legacy_python or _legacy_executable(protocol)
    host = legacy_host if legacy_host is not None else protocol["provenance"]["dependencies"].get("hostname")
    command = [python, "-B", str(Path(__file__).resolve()), "--audit-legacy",
               "--study-dir", str(Path(study_dir).resolve()), "--baseline-dir", str(Path(baseline_dir).resolve())]
    if host and host != socket.gethostname():
        if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
            raise ValueError("Invalid legacy audit hostname")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(command)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise ValueError("Legacy baseline schedule audit failed: " + result.stderr[-2000:] + result.stdout[-500:])
    try:
        audited = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("Legacy audit returned invalid structured evidence") from exc
    if audited.get("status") != "verified" or audited.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Legacy audit certificate does not match the frozen baseline protocol")
    return audited


def _audit_in_legacy(study_dir, baseline_dir):
    """Re-execute original environment semantics; read files but load no policy."""
    study, protocol, receipt, identities = _verify_completed_artifacts(study_dir, baseline_dir)
    spec = importlib.util.spec_from_file_location("phi3_legacy_baseline_audit", Path(baseline_dir) / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize_runtime()
    checked_protocol, rows = module.verify(Path(baseline_dir) / "artifacts/PROTOCOL.json")
    recalculated = module.summarize(checked_protocol, rows, Path(baseline_dir) / "artifacts/production", PURPOSE)
    ignored = {"completed_at_utc", "receipt_sha256"}
    if {key: value for key, value in recalculated.items() if key not in ignored} != {key: value for key, value in receipt.items() if key not in ignored}:
        raise ValueError("Completed baseline receipt differs from fresh legacy-environment revalidation")
    candidates = {}
    for name in MODELS:
        candidates[name] = []
        for identity in identities:
            path = Path(baseline_dir) / "artifacts/production" / name / f"{identity}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                # summarize just independently reconstructed and revalidated
                # every row, including L2D's left-shift environment replay.
                validation = {key: value for key, value in row["validation"].items() if key != "operations"}
                candidates[name].append({"instance_id": identity, "candidate_id": row["candidate_index"],
                                          "validation": validation})
    return {"status": "verified", "protocol_sha256": protocol["protocol_sha256"],
            "receipt_sha256": receipt["receipt_sha256"], "candidates": candidates,
            "certificate": {"legacy_executable": sys.executable, "hostname": socket.gethostname(),
                "legacy_fingerprints_verified_in_legacy_environment": True,
                "raw_schedule_reconstruction_verified": True, "receipt_recomputed": True,
                "instances": 216, "candidates_per_instance": 10, "models": 4}}


def _population(records, identities, label):
    expected_ids = set(identities)
    slots = set()
    for record in records:
        key = record.instance_id, record.candidate_id
        if record.instance_id not in expected_ids or type(record.candidate_id) is not int or not 0 <= record.candidate_id < 10 or key in slots:
            raise ValueError(f"{label}: wrong or duplicate candidate identity")
        slots.add(key)
        gap = record.validation.gap_percent
        if record.validation.feasible:
            if gap is None or not math.isfinite(gap):
                raise ValueError(f"{label}: feasible candidate lacks a finite reference gap")
        elif gap is not None:
            raise ValueError(f"{label}: infeasible candidate has a conditional gap")
    if slots != {(identity, candidate) for identity in identities for candidate in range(10)}:
        raise ValueError(f"{label}: incomplete fixed candidate population")
    metric = aggregate_results(records, ks=(10,))
    best_gaps = _best_gaps(records)
    metric["observed_max_best_of_10_gap_percent"] = max(best_gaps.values()) if best_gaps else None
    return metric


def _best_gaps(records):
    result = {}
    for record in records:
        if record.validation.feasible:
            gap = record.validation.gap_percent
            result[record.instance_id] = min(gap, result.get(record.instance_id, gap))
    return result


def _gap_stats(values):
    values = list(values)
    return {"mean_percent": statistics.mean(values) if values else None,
            "median_percent": statistics.median(values) if values else None}


def _paired_gap_summary(phi_records, baseline_records):
    phi, baseline = _best_gaps(phi_records), _best_gaps(baseline_records)
    target = sorted(phi)
    missing = [identity for identity in target if identity not in baseline]
    complete = not missing
    # Never silently drop a baseline failure from the requested Phi-feasible
    # subset. The full-population feasibility metric still counts that failure.
    paired = target if complete else []
    return {"subset": "identical Phi-feasible instances; best valid candidate among ten for each method",
            "n_phi_feasible_instances": len(target), "n_paired_instances": len(paired),
            "n_baseline_infeasible_on_phi_subset": len(missing), "subset_complete": complete,
            "missing_baseline_instance_ids": missing,
            "phi_gap": _gap_stats(phi[identity] for identity in paired),
            "baseline_gap": _gap_stats(baseline[identity] for identity in paired),
            "baseline_minus_phi_gap_percentage_points": _gap_stats(baseline[identity] - phi[identity] for identity in paired)}


def _number(value):
    return "--" if value is None else f"{value:.2f}"


def _tex(metrics, phi_metrics, paired, selected):
    full_rows = []
    for name, metric in list(metrics.items()) + [("phi:" + key, value) for key, value in phi_metrics.items()]:
        label = MODEL_LABELS[name] if name in MODEL_LABELS else "Phi-3 " + name[4:].replace("_", r"\_")
        best = metric["best_of_k"]["10"]
        gaps = best["conditional_best_gap"]
        full_rows.append(f"{label} & {best['instances_with_feasible_candidate']}/216 & {_number(100*best['instance_feasibility_rate'])} & {_number(gaps['mean_gap_percent'])} & {_number(gaps['median_gap_percent'])} & {_number(metric['observed_max_best_of_10_gap_percent'])}" + r" \\" + "\n")
    paired_rows = []
    for prompt, by_model in paired.items():
        for name, value in by_model.items():
            delta = value["baseline_minus_phi_gap_percentage_points"]
            prompt_label = prompt.replace("_", r"\_")
            paired_rows.append(f"{prompt_label} & {MODEL_LABELS[name]} & {value['n_paired_instances']}/{value['n_phi_feasible_instances']} & {_number(value['baseline_gap']['mean_percent'])} & {_number(value['phi_gap']['mean_percent'])} & {_number(delta['mean_percent'])}" + r" \\" + "\n")
    return (
        r"\subsubsection{Supplementary comparisons on the frozen test population}" + "\n"
        "We additionally evaluate four released specialized policies on the same 216 instances, retaining exactly ten sampled schedules per instance and method. "
        "All raw schedules are independently reconstructed and strictly revalidated, including the SLJ decoder's forced final operation and L2D's native permissible left shift. "
        "Reference gaps use the released feasible solver labels; their optimality is not assumed. The first table retains each method's full-population feasibility and conditional best-of-ten gaps. "
        "Its observed maximum is the largest per-instance best-valid-of-ten gap among that method's feasible instances; this conditional sample maximum is not a worst-case guarantee.\n\n"
        r"\begin{table}[!ht]\centering\small\caption{Supplementary comparison with released policies. Feasibility uses all 216 instances; mean, median, and observed maximum reference gaps are conditional on each method's own feasible subset and use its best valid candidate among ten. The observed maximum is not a worst-case guarantee. Equal candidate counts do not imply equal compute.}\label{tab:phi3_supplementary_baselines}" + "\n"
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrr}\toprule Method & Feasible & Feasibility (\%) & Mean gap (\%) & Median gap (\%) & Observed max gap (\%) \\ \midrule" + "\n"
        + "".join(full_rows) + r"\bottomrule\end{tabular}}\end{table}" + "\n\n"
        "To avoid comparing gaps computed on different feasible populations, the next table restricts both schedules to the identical subset where the indicated Phi-3 instruction generated a valid schedule. "
        "Each entry uses the best valid schedule among ten for both methods. If a baseline fails on any instance in that Phi-feasible subset, its paired gap entry is withheld rather than silently dropping that instance. "
        "The reported difference is the baseline reference gap minus the Phi-3 reference gap, in percentage points; these are descriptive paired comparisons.\n\n"
        r"\begin{table}[!ht]\centering\small\caption{Paired reference gaps on each indicated Phi-3-feasible subset. Coverage gives paired instances over all Phi-feasible instances.}\label{tab:phi3_baseline_paired_gaps}" + "\n"
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrr}\toprule Phi instruction & Baseline & Coverage & Baseline mean (\%) & Phi mean (\%) & Mean difference (pp) \\ \midrule" + "\n"
        + "".join(paired_rows) + r"\bottomrule\end{tabular}}\end{table}" + "\n\n"
        "The validation-selected Phi-3 instruction is chosen using the separate validation-only selector; no baseline outcomes determine this choice. "
        "All policies run in evaluation mode; the released L2D testing script omits this call, so these results explicitly use frozen BatchNorm statistics. "
        "The released SLJ B256 decoder has three attention heads, whereas B32 and B128 have four; these checkpoints are different released configurations, rather than a controlled ablation of the training sampling parameter alone. "
        "The policies have different training histories, architectures, and execution costs. Baselines run on one CPU thread and SLJ samples ten trajectories in one batch; their CPU scheduling and feature times are retained separately. "
        "Feasibility and the paired comparisons must accompany conditional gaps; gaps from different feasible subsets do not establish overall superiority.\n"
    )


def build_comparison(study_dir, runs_dir, baseline_dir, phi_records_by_prompt,
                     phi_summary_by_prompt, selected_prompt_id, legacy_python=None, legacy_host=None):
    """Return TeX and measured summaries; never edit a manuscript or artifact."""
    study, protocol, receipt, identities = _verify_completed_artifacts(study_dir, baseline_dir)
    selection_path = Path(study_dir) / "selected_prompt.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("selected_prompt_id") != selected_prompt_id or selection.get("study_protocol_sha256") != study["protocol_sha256"]:
        raise ValueError("Requested Phi instruction differs from the frozen validation selection")
    final_manifest = Path(runs_dir) / "evaluations/final_prompt_test/evaluation_manifest.json"
    if not final_manifest.is_file() or selection_path.stat().st_mtime > final_manifest.stat().st_mtime:
        raise ValueError("Phi instruction was not frozen before final test generation")
    prompts = list(dict.fromkeys(("seen_1", selected_prompt_id)))
    phi_metrics = {}
    for prompt in prompts:
        if prompt not in phi_records_by_prompt or prompt not in phi_summary_by_prompt:
            raise ValueError("Verified Phi records and summaries are required for both comparison instructions")
        metric = _population(phi_records_by_prompt[prompt], identities, "Phi " + prompt)
        supplied = phi_summary_by_prompt[prompt]
        for key in ("n_instances", "n_candidates", "feasible_candidates", "instances_with_any_feasible_candidate"):
            if metric[key] != supplied[key]:
                raise ValueError("Supplied Phi summary disagrees with its verified raw records")
        for key in ("n_eligible_instances", "instances_with_feasible_candidate", "instance_feasibility_rate", "conditional_best_gap"):
            if metric["best_of_k"]["10"][key] != supplied["best_of_k"]["10"][key]:
                raise ValueError("Supplied Phi best-of-ten gaps disagree with verified raw records")
        phi_metrics[prompt] = metric
    audited = _legacy_audit(study_dir, baseline_dir, protocol, legacy_python, legacy_host)
    if audited.get("receipt_sha256") != receipt["receipt_sha256"] or set(audited.get("candidates", {})) != set(MODELS):
        raise ValueError("Legacy audited evidence does not match the completed baseline receipt")
    baseline_records, metrics = {}, {}
    for name in MODELS:
        records = []
        for row in audited["candidates"][name]:
            value = row["validation"]
            result = ValidationResult(value["feasible"], tuple(value["errors"]), tuple(value["details"]), (),
                                      value["reported_makespan"], value["computed_makespan"], value["gap_percent"])
            records.append(EvaluationRecord(row["instance_id"], row["candidate_id"], result))
        baseline_records[name] = records
        metrics[name] = _population(records, identities, name)
    paired = {prompt: {name: _paired_gap_summary(phi_records_by_prompt[prompt], records)
                       for name, records in baseline_records.items()} for prompt in prompts}
    structured = {"status": "verified_supplementary_comparison", "instances": 216, "candidates_per_instance": 10,
                  "baseline_protocol_sha256": protocol["protocol_sha256"], "baseline_receipt_sha256": receipt["receipt_sha256"],
                  "parent_study_protocol_sha256": study["protocol_sha256"], "legacy_audit": audited["certificate"],
                  "selected_prompt_id": selected_prompt_id, "baseline_metrics": metrics, "phi_metrics": phi_metrics,
                  "observed_maximum_definition": "maximum per-instance best valid candidate gap among ten, conditional on that method's feasible subset; descriptive sample maximum, not a worst-case guarantee",
                  "paired_reference_gaps_on_phi_feasible_subsets": paired,
                  "baseline_cpu_times": {name: {key: receipt["models"][name][key] for key in ("scheduling_seconds", "feature_seconds")} for name in MODELS},
                  "caveats": {key: protocol[key] for key in ("evaluation_mode", "batching", "compute_caveat", "checkpoint_training_provenance")},
                  "slj_architecture_caveat": protocol["slj"]["architecture_caveat"],
                  "l2d_evaluation_mode_difference": protocol["l2d"]["mode_difference"]}
    return {"tex": _tex(metrics, phi_metrics, paired, selected_prompt_id), "summary": structured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-legacy", action="store_true", required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_audit_in_legacy(args.study_dir, args.baseline_dir), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
