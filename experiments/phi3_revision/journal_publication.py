#!/usr/bin/env python3
"""Guarded, presentation-only bridge from frozen results to the journal paper.

Uses the standard library and the pure journal table reflow helper. It does not
load models, start jobs, access a network, compile PDFs, or repeat model forwards.
The frozen scientific reporters remain responsible for raw schedule revalidation;
this bridge checks their completed publication chain and bound result artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

try:
    from .journal_tables import reflow_result_tables
except ImportError:
    from journal_tables import reflow_result_tables

MAIN_PROTOCOL = "dc0daecee91f65d3bc6204c4031d509b027161350637a00834a6785757f51616"
AUTO_PROTOCOL = "2a6b4408c73a7c5f6a93be3500d7af1aa07279717ea05941db55e8d6b8bb4479"
BASELINE_PROTOCOL = "a9a248f046260ca5fa04c387e327420095059f19f9ca6ca834c70e0590863bb3"
PROMPT_IDS = ("seen_1", "seen_2", "seen_3", "unseen_1", "unseen_2", "unseen_3", "unseen_4")
AUTO_IDS = ("auto_0", "auto_r1_s1", "auto_r1_s2", "auto_r2_s1", "auto_r2_s2")
TARGETS = ("journal_abstract.tex", "journal_intro.tex", "journal_conclusion.tex",
           "journal_results.tex", "journal_status.tex")
RECEIPT = "JOURNAL_RESULT_INTEGRATION.json"
PRECISION_MARKER = "% BEGIN VERIFIED JOURNAL PRECISION CLARIFICATION"


class PublicationGuardError(ValueError):
    pass


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise PublicationGuardError(message)


def load_json(path, bindings, self_key=None):
    path = Path(path).resolve()
    require(path.is_file(), f"Required completion/evidence file is absent: {path}")
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise PublicationGuardError(f"Unreadable evidence: {path}") from exc
    require(isinstance(value, dict), f"Expected evidence object: {path}")
    bindings[str(path)] = file_hash(path)
    if self_key:
        require(value.get(self_key) == json_hash({k: v for k, v in value.items() if k != self_key}),
                f"Invalid {self_key}: {path}")
    return value


def bound_file(path, expected, bindings):
    path = Path(path).resolve()
    require(isinstance(expected, str) and bool(re.fullmatch("[0-9a-f]{64}", expected)),
            f"Missing/invalid expected content hash for {path}")
    require(path.is_file() and file_hash(path) == expected, f"Bound artifact changed: {path}")
    bindings[str(path)] = expected
    return path


def _safe_relative(directory, name):
    path = (Path(directory) / name).resolve()
    require(path.is_relative_to(Path(directory).resolve()), "Artifact path escapes its receipt directory")
    return path


def _summary(summary):
    require(summary.get("n_instances") == 216 and summary.get("n_candidates") == 2160,
            "A result does not contain the complete 216 x 10 population")
    best = summary.get("best_of_k", {}).get("10", {})
    count = best.get("instances_with_feasible_candidate")
    valid = summary.get("feasible_candidates")
    require(type(count) is int and 0 <= count <= 216 and best.get("n_eligible_instances") == 216,
            "Invalid best-of-ten instance denominator")
    require(type(valid) is int and count <= valid <= 2160, "Invalid candidate feasibility count")
    require(summary.get("instances_with_any_feasible_candidate") == count,
            "Instance feasibility summaries disagree")
    gap = best.get("conditional_best_gap", {})
    for key in ("mean_gap_percent", "median_gap_percent"):
        value = gap.get(key)
        require(value is None or (isinstance(value, (int, float)) and math.isfinite(value)),
                "Nonfinite conditional reference gap")
        require(count != 0 or value is None, "A zero-feasibility result cannot have a gap")
    return count, gap.get("mean_gap_percent")


def _rank(scores, order):
    def key(identity):
        score = scores[identity]
        valid = score.get("feasible_candidates")
        gap = score.get("conditional_mean_gap_percent")
        require(type(valid) is int and 0 <= valid <= 216, "Invalid validation candidate count")
        require(gap is None or (isinstance(gap, (int, float)) and math.isfinite(gap)),
                "Nonfinite validation reference gap")
        require(valid != 0 or gap is None, "Undefined validation gap must remain null")
        return -valid, math.inf if gap is None else gap, order.index(identity)
    require(set(scores) == set(order), "Selection scores do not match the complete registry")
    return min(order, key=key)


def _evaluation_receipt(folder, expected, bindings):
    names = {"evaluation_manifest_sha256": "evaluation_manifest.json",
             "candidates_sha256": "candidates.jsonl", "complete_sha256": "COMPLETE",
             "instruction_manifest_sha256": "instruction_manifest.json"}
    require(set(expected) == set(names), "Incomplete dynamic-evaluation receipt")
    for key, name in names.items():
        bound_file(Path(folder) / name, expected[key], bindings)


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise PublicationGuardError("Invalid evidence timestamp") from exc
    require(result.tzinfo is not None, "Evidence timestamps must include a timezone")
    return result


def _primary_evaluation(folder, ids, prompts, candidates, registry, data_path, adapter, bindings):
    folder = Path(folder)
    manifest = load_json(folder / "evaluation_manifest.json", bindings, "configuration_sha256")
    marker = load_json(folder / "COMPLETE", bindings)
    summary = load_json(folder / "summary.json", bindings)
    expected = len(ids) * len(prompts) * candidates
    config = manifest["config"]
    require(manifest.get("run_kind") == "FULL" and manifest.get("instance_ids") == ids
            and manifest.get("frozen_prompts") == registry["prompts"]
            and manifest.get("expected_candidate_slots") == expected
            and manifest.get("evaluated_instances") == len(ids)
            and config.get("limit") is None and config.get("prompt_ids") == list(prompts)
            and config.get("candidate_count") == candidates
            and config.get("batch_size") == (3 if candidates == 3 else 5)
            and (config.get("temperature"), config.get("top_k"), config.get("top_p"), config.get("seed")) == (1.0, 50, 0.95, 42)
            and Path(config["adapter"]).resolve() == Path(adapter).resolve(),
            "Primary evaluation population, registry, model, or decoding differs")
    bound_file(data_path, manifest.get("source_data_sha256"), bindings)
    require(marker.get("status") == summary.get("status") == "complete"
            and marker.get("configuration_sha256") == summary.get("configuration_sha256") == manifest["configuration_sha256"]
            and marker.get("candidate_slots") == summary.get("expected_candidate_slots") == summary.get("terminal_candidate_slots") == expected
            and summary.get("candidate_status_counts") == {"generated": expected}
            and summary.get("not_generated_slots") == 0, "Primary evaluation has incomplete/skipped slots")
    raw = folder / "candidates.jsonl"
    require(raw.is_file(), "Primary evaluation lacks retained raw candidates")
    bindings[str(raw.resolve())] = file_hash(raw)
    return manifest, summary


def verify_publication_chain(paper_dir, study_dir, runs_dir, auto_protocol_path, publisher_receipt_path):
    """Read all barriers and receipt links before creating any path or file."""
    paper, study_dir, runs = map(lambda p: Path(p).resolve(), (paper_dir, study_dir, runs_dir))
    bindings = {}
    # These are checked first: an incomplete run leaves even build/ untouched.
    train = load_json(runs / "main/training_complete.json", bindings)
    manager = load_json(runs / "manager/COMPLETE", bindings)
    protocol = load_json(auto_protocol_path, bindings, "protocol_sha256")
    output = Path(protocol["output_dir"]).resolve()
    scientific = load_json(output / "COMPLETE.json", bindings, "receipt_sha256")
    integrated = load_json(output / "INTEGRATED.json", bindings, "receipt_sha256")
    published = load_json(publisher_receipt_path, bindings, "receipt_sha256")
    study = load_json(study_dir / "STUDY.json", bindings, "protocol_sha256")
    require(study["protocol_sha256"] == MAIN_PROTOCOL and protocol["protocol_sha256"] == AUTO_PROTOCOL,
            "Bridge accepts only the reviewed frozen primary and automatic studies")
    require(protocol.get("parent_protocol_sha256") == MAIN_PROTOCOL
            and Path(protocol["parent_study"]).resolve() == study_dir / "STUDY.json"
            and Path(protocol["runs_dir"]).resolve() == runs
            and Path(protocol["training_complete"]).resolve() == runs / "main/training_complete.json",
            "Automatic study points to another primary run")
    require(train.get("global_step") == 1844 and not train.get("pilot"), "Full non-pilot training is required")
    require(manager == {"status": "complete", "study_protocol_sha256": MAIN_PROTOCOL},
            "Primary manager has not completed this frozen study")
    run = load_json(runs / "main/run_manifest.json", bindings)
    require(run.get("status") == "complete" and run.get("config", {}).get("total_steps") == 1844
            and not run["config"].get("pilot") and run["train_data"]["records"] == 118000
            and run["validation_data"]["records"] == 2000, "Training manifest is incomplete or has another population")
    status = load_json(runs / "manager/status.json", bindings)
    require(status.get("status") == "complete" and status.get("training", {}).get("status") == "complete"
            and status.get("report") == status.get("pdf") == "complete"
            and set(status.get("jobs", {})) == {"checkpoint_0", "checkpoint_461", "checkpoint_922", "checkpoint_1383", "final_prompt_validation", "final_prompt_test"}
            and all(job.get("status") == "complete" for job in status["jobs"].values()),
            "Primary manager lacks completed required jobs, report, or PDF")
    for step in (0, 461, 922, 1383, 1844):
        cp = load_json(runs / f"main/checkpoint-{step}/experiment_checkpoint.json", bindings)
        require(cp.get("global_step") == step, "Checkpoint completion marker has the wrong step")
    require(study.get("test_count") == 216 and study.get("prompt_validation_count") == 72
            and study.get("checkpoint_steps") == [0, 461, 922, 1383, 1844], "Frozen populations/checkpoints differ")
    for name, digest in study["artifact_sha256"].items():
        bound_file(_safe_relative(study_dir, name), digest, bindings)
    registry = load_json(study_dir / "PROMPTS.json", bindings)
    require(json_hash(registry) == study["prompt_registry_sha256"]
            and tuple(p["id"] for p in registry["prompts"]) == PROMPT_IDS
            and [p["seen_in_training"] for p in registry["prompts"]] == [True] * 3 + [False] * 4,
            "The seven-string frozen registry changed")
    require(protocol.get("test_instance_ids") == study["test_instance_ids"]
            and protocol.get("validation_instance_ids") == study["prompt_validation_instance_ids"]
            and len(set(study["test_instance_ids"])) == 216
            and len(set(study["prompt_validation_instance_ids"])) == 72
            and not set(study["test_instance_ids"]) & set(study["prompt_validation_instance_ids"]),
            "Primary/automatic populations differ or overlap")
    for name, digest in protocol["source_sha256"].items():
        bound_file(name, digest, bindings)
    require(scientific.get("status") == "complete" and scientific.get("purpose") == "SCIENTIFIC_AUTO_PROMPT"
            and scientific.get("protocol_sha256") == AUTO_PROTOCOL
            and (scientific.get("validation_schedules"), scientific.get("test_schedules")) == (1080, 2160),
            "Automatic scientific completion is incomplete, a pilot, or has another budget")
    auto_run = load_json(output / "RUN.json", bindings, "run_sha256")
    require(auto_run.get("protocol_sha256") == AUTO_PROTOCOL
            and auto_run.get("parent_protocol_sha256") == MAIN_PROTOCOL
            and auto_run.get("purpose") == "SCIENTIFIC_AUTO_PROMPT", "Automatic RUN receipt belongs to another study")
    for field, path in (("primary_manager_complete_sha256", runs / "manager/COMPLETE"),
                        ("training_complete_sha256", runs / "main/training_complete.json"),
                        ("training_manifest_sha256", runs / "main/run_manifest.json")):
        bound_file(path, auto_run.get(field), bindings)
    selection = load_json(output / "SELECTED.json", bindings, "selection_sha256")
    require(selection.get("status") == "frozen_before_auto_test"
            and selection.get("protocol_sha256") == AUTO_PROTOCOL
            and selection["selection_sha256"] == scientific.get("selection_sha256"), "Automatic selection binding differs")
    require(selection.get("selected_instruction_id") == scientific.get("test_instruction_id")
            == _rank(selection["scores"], AUTO_IDS), "Automatic selection does not follow validation-only ranking")
    require(set(selection["validation_receipts"]) == set(AUTO_IDS), "Missing automatic validation receipt")
    for identity in AUTO_IDS:
        feedback = selection["scores"][identity]
        require(feedback.get("split") == "validation" and (feedback.get("n_instances"), feedback.get("candidate_count"), feedback.get("n_candidates")) == (72, 3, 216)
                and feedback.get("instance_ids_sha256") == json_hash(protocol["validation_instance_ids"])
                and not feedback.get("pilot_excluded"), "Automatic selection used a different validation population/budget")
        _evaluation_receipt(output / "evaluations/validation" / identity, selection["validation_receipts"][identity], bindings)
    auto_id = selection["selected_instruction_id"]
    _evaluation_receipt(output / "evaluations/test" / auto_id, scientific["test_receipt"], bindings)
    test_instruction = load_json(output / "evaluations/test" / auto_id / "instruction_manifest.json", bindings, "manifest_sha256")
    require(test_instruction.get("selected_before_test_sha256") == selection["selection_sha256"]
            and _time(selection["selected_at_utc"]) <= _time(test_instruction["created_at_utc"])
            and test_instruction.get("instruction") == selection["selected_instruction"]
            and test_instruction.get("split") == "test" and not test_instruction.get("pilot_excluded"),
            "Automatic test began before frozen selection or used a different instruction")
    spec = selection["selected_instruction"]
    require(spec.get("id") == auto_id and spec.get("text_sha256") == hashlib.sha256(spec["text"].encode()).hexdigest(),
            "Selected instruction text/identity hash differs")
    for name, digest in scientific["artifact_sha256"].items():
        bound_file(_safe_relative(output, name), digest, bindings)
    primary = load_json(study_dir / "integrated_results.json", bindings)
    require(primary.get("status") == "verified_results_integrated"
            and primary.get("report_code_sha256") == protocol["source_sha256"][str((runs.parent / "report.py").resolve())],
            "Primary measured integration is absent or uses another reporter")
    bound_file(paper / "revision_results.tex", primary.get("results_tex_sha256"), bindings)
    bound_file(paper / "build/neurips_2024.before_auto_prompt_integration.tex", primary.get("tex_sha256"), bindings)
    require(set(primary.get("checkpoint_summaries", {})) == {"0", "461", "922", "1383", "1844"}, "Missing measured training-budget rows")
    for summary in primary["checkpoint_summaries"].values():
        _summary(summary)
    require(set(primary.get("prompt_summary", {}).get("by_prompt", {})) == set(PROMPT_IDS), "Missing measured seven-prompt results")
    for summary in primary["prompt_summary"]["by_prompt"].values():
        _summary(summary)
    require(primary["checkpoint_summaries"]["1844"] == primary["prompt_summary"]["by_prompt"]["seen_1"], "Final checkpoint and original-instruction results disagree")
    parent_selection = load_json(study_dir / "selected_prompt.json", bindings)
    require(parent_selection == primary.get("selected_prompt")
            and parent_selection.get("study_protocol_sha256") == MAIN_PROTOCOL
            and parent_selection.get("selection_split") == "validation"
            and parent_selection.get("n_validation_instances") == 72
            and parent_selection.get("selected_prompt_id") == _rank(parent_selection["scores"], PROMPT_IDS),
            "Fixed-registry selection is not supported by validation-only evidence")
    supplementary = primary.get("supplementary_baselines")
    require(isinstance(supplementary, dict)
            and supplementary.get("status") == "verified_supplementary_comparison"
            and supplementary.get("instances") == 216 and supplementary.get("candidates_per_instance") == 10
            and supplementary.get("parent_study_protocol_sha256") == MAIN_PROTOCOL
            and supplementary.get("selected_prompt_id") == parent_selection["selected_prompt_id"],
            "Journal baseline comparison requires its completed matched supplement")
    baseline_dir = runs.parent / "baselines/artifacts"
    bp = load_json(baseline_dir / "PROTOCOL.json", bindings, "protocol_sha256")
    bc = load_json(baseline_dir / "production/COMPLETE.json", bindings, "receipt_sha256")
    require(bp["protocol_sha256"] == BASELINE_PROTOCOL
            and bp.get("parent_protocol_sha256") == MAIN_PROTOCOL
            and bc.get("protocol_sha256") == supplementary.get("baseline_protocol_sha256") == BASELINE_PROTOCOL
            and bc.get("receipt_sha256") == supplementary.get("baseline_receipt_sha256")
            and bc.get("status") == "complete" and bc.get("purpose") == "SCIENTIFIC_FROZEN_216"
            and bc.get("instances") == 216 and bc.get("candidates_per_instance") == 10
            and set(bc.get("models", {})) == {"l2d_20x20_1_99", "slj_b32", "slj_b128", "slj_b256"},
            "Matched supplementary baseline receipt differs")
    for model in bc["models"].values():
        require(model.get("candidates") == 2160 and len(model.get("raw_file_sha256", {})) == 216,
                "A baseline arm lacks its full fixed candidate population")
        for name, digest in model["raw_file_sha256"].items():
            bound_file(_safe_relative(baseline_dir / "production", name), digest, bindings)
    bound_file(runs / "evaluations/final_prompt_validation/candidates.jsonl", parent_selection["validation_candidates_sha256"], bindings)
    vm, validation_summary = _primary_evaluation(runs / "evaluations/final_prompt_validation",
        study["prompt_validation_instance_ids"], PROMPT_IDS, 3, registry, study_dir / "prompt_validation.jsonl",
        runs / "main/checkpoint-1844", bindings)
    fm, final_summary = _primary_evaluation(runs / "evaluations/final_prompt_test",
        study["test_instance_ids"], PROMPT_IDS, 10, registry, study_dir / "test.jsonl",
        runs / "main/checkpoint-1844", bindings)
    require(final_summary == primary["prompt_summary"]
            and parent_selection.get("validation_configuration_sha256") == vm["configuration_sha256"]
            and parent_selection.get("prompt_registry_sha256") == study["prompt_registry_sha256"]
            and parent_selection.get("study_test_sha256") == fm["source_data_sha256"]
            and (study_dir / "selected_prompt.json").stat().st_mtime <= (runs / "evaluations/final_prompt_test/evaluation_manifest.json").stat().st_mtime,
            "Fixed-registry selection/evaluation provenance or chronology differs")
    for identity in PROMPT_IDS:
        value = validation_summary["by_prompt"][identity]
        expected_score = {"n_candidates": 216, "feasible_candidates": value["feasible_candidates"],
                          "conditional_mean_gap_percent": value["conditional_candidate_gap"]["mean_gap_percent"]}
        require(parent_selection["scores"][identity] == expected_score, "Fixed-registry selection score differs from validation summary")
    for step in (0, 461, 922, 1383):
        _, measured = _primary_evaluation(runs / f"evaluations/checkpoint_{step}", study["test_instance_ids"],
            ("seen_1",), 10, registry, study_dir / "test.jsonl", runs / f"main/checkpoint-{step}", bindings)
        require(measured["by_prompt"]["seen_1"] == primary["checkpoint_summaries"][str(step)], "Measured checkpoint summary differs from integrated row")
    require(integrated.get("status") == "verified_auto_prompt_results"
            and integrated.get("protocol_sha256") == AUTO_PROTOCOL
            and integrated.get("complete_receipt_sha256") == scientific["receipt_sha256"]
            and (integrated.get("validation_candidates"), integrated.get("test_candidates")) == (1080, 2160)
            and integrated.get("selected_instruction") == selection["selected_instruction"]
            and integrated.get("validation_scores") == selection["scores"]
            and integrated.get("parent_selected_prompt") == parent_selection["selected_prompt_id"],
            "Automatic measured integration does not bind the completed scientific run")
    _summary(integrated["test_summary"])
    bound_file(paper / "auto_prompt_results.tex", integrated["results_tex_sha256"], bindings)
    require(published.get("status") == "complete" and published.get("purpose") == "VERIFIED_AUTO_PROMPT_PUBLICATION"
            and published.get("protocol_sha256") == AUTO_PROTOCOL
            and published.get("scientific_receipt_sha256") == scientific["receipt_sha256"]
            and published.get("integrated_receipt_sha256") == integrated["receipt_sha256"],
            "Frozen publishing driver has not completed this integrated scientific result")
    bound_file(paper / "neurips_2024.tex", published.get("paper_source_sha256"), bindings)
    require(integrated.get("paper_sha256") == published["paper_source_sha256"]
            and integrated.get("rebuttal_sha256") == published.get("rebuttal_source_sha256"), "Integrated sources and published sources disagree")
    prior_path = paper / "build" / RECEIPT
    prior = load_json(prior_path, bindings, "receipt_sha256") if prior_path.exists() else None
    rebuttal_source = paper / "rebuttal.tex"
    if prior:
        require(prior.get("status") == "verified_journal_results_integrated_pdf_rebuild_required"
                and prior.get("publisher_receipt_file_sha256") == file_hash(publisher_receipt_path)
                and prior.get("source_code_sha256") == file_hash(__file__)
                and prior.get("table_reflow_code_sha256") == file_hash(Path(__file__).with_name("journal_tables.py")),
                "Existing journal bridge belongs to another publication or implementation")
        for name, digest in prior["outputs_sha256"].items():
            bound_file(paper / name, digest, bindings)
        rebuttal_source = Path(prior["backups"]["rebuttal.tex"])
    bound_file(rebuttal_source, published["rebuttal_source_sha256"], bindings)
    evidence = Path(published["evidence_dir"]).resolve()
    bound_file(evidence / "COMPLETE.json", file_hash(publisher_receipt_path), bindings)
    launch = load_json(evidence / "LAUNCH.json", bindings)
    for name in ("neurips_2024", "rebuttal"):
        check = published["pdfs"][name]
        require(all(check.get(k) is True for k in ("actual_title_present", "formatting_sample_absent", "measured_sections_and_tables_present", "verified_headline_present", "all_pages_inspected", "no_tex_errors", "references_resolved", "labels_stable", "no_missing_characters", "no_overfull_boxes")), "Publisher PDF inspection is incomplete")
        bound_file(Path(launch["build_dir"]) / (name + ".pdf"), check["pdf_sha256"], bindings)
        if not prior:
            bound_file(paper / (name + ".pdf"), check["pdf_sha256"], bindings)
    precision_path = runs.parent / "artifacts/PRECISION_CLARIFICATION.json"
    precision = load_json(precision_path, bindings, "clarification_sha256")
    require(precision.get("purpose") == "RETROSPECTIVE_INFERENCE_PRECISION_DESCRIPTION_CLARIFICATION"
            and precision.get("no_experiment_or_frozen_artifact_changed") is True, "Precision clarification lacks its audited scope")
    for identity in ("main_protocol", "auto_protocol", "precision_audit", "checkpoint0_initialization_audit", "publication_driver"):
        item = precision["bindings"][identity]
        bound_file(item["path"], item["file_sha256"], bindings)
    require(precision["bindings"]["main_protocol"]["protocol_sha256"] == MAIN_PROTOCOL
            and precision["bindings"]["auto_protocol"]["protocol_sha256"] == AUTO_PROTOCOL
            and precision["bindings"]["publication_driver"]["file_sha256"] == published["driver_sha256"], "Precision correction refers to another protocol/driver")
    return dict(paper=paper, primary=primary, automatic=integrated, precision=precision,
                published=published, prior=prior, bindings=bindings)


def _escape(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(ch, ch) for ch in text)


def _outcome(label, summary):
    count, mean = _summary(summary)
    sentence = f"{label} yields a valid schedule for {count}/216 instances ({100 * count / 216:.2f}\\%)"
    if mean is not None:
        return sentence + f", with a conditional mean reference gap of {mean:.2f}\\%."
    if count == 0:
        return sentence + "; its conditional gap is undefined because no valid schedule was found."
    return sentence + "; a conditional reference-gap estimate is unavailable."


def transform_results(primary_tex, automatic_tex, precision):
    text = primary_tex.rstrip() + "\n\n" + automatic_tex.rstrip() + "\n"
    correction = precision["publication_correction"]
    text = text.replace(correction["exact_fresh_evaluation_sentence_to_replace"], correction["exact_replacement"])
    text = text.replace(r"\subsubsection{", r"\subsection{")
    # Generated paragraph titles contain no nested commands/braces; refuse a
    # changed format instead of silently corrupting a future heading.
    text = re.sub(r"\\paragraph\{([^{}]+)\}", lambda m: r"\noindent\textbf{" + m[1].rstrip(".") + ".} ", text)
    require(r"\paragraph{" not in text and not re.search(r"(?m)^\\section\{", text), "Unexpected generated heading hierarchy")
    return (r"\section{Results}\label{sec:journal_results}" + "\n\n"
            + r"\input{journal_original_comparison}" + "\n\n" + reflow_result_tables(text))


def render_fragments(primary, automatic, primary_tex, automatic_tex, precision):
    """Pure measured rendering, including zero/null/negative-gap outcomes."""
    original = primary["checkpoint_summaries"]["1844"]
    fixed_id = primary["selected_prompt"]["selected_prompt_id"]
    require(fixed_id in PROMPT_IDS, "Unknown fixed-registry selected instruction")
    auto_id = automatic["selected_instruction"]["id"]
    require(auto_id in AUTO_IDS, "Unknown automatic selected instruction")
    first = _outcome("The original instruction", original)
    fixed = _outcome("The validation-selected fixed instruction", primary["prompt_summary"]["by_prompt"][fixed_id])
    generated = _outcome("The separately selected refinement instruction", automatic["test_summary"])
    abstract = (
        "We study whether supervised fine-tuning enables a large language model to generate complete schedules for the classical job shop scheduling problem. "
        "We provide 120,000 natural-language instances paired with feasible solver schedules, using 118,000 for training and 2,000 for validation. "
        "A single-seed Phi-3-Mini-128K-Instruct run is evaluated at five checkpoints on 216 frozen test instances with exactly ten candidates per instance and strict schedule validation. "
        + first + " Prompt selection uses a frozen 72-instance validation subset, with two instances from each of 36 size groups. "
        + fixed + " A separate two-round procedure proposes four new instructions using aggregate validation feedback and freezes its selection before fresh testing. "
        + generated + " Released specialized policies are compared on the same test population and candidate budget, with paired quality comparisons on identical feasible subsets. "
        "Reference gaps are conditional on feasibility and do not certify optimality. Equal candidate counts do not imply equal compute. "
        "The study measures instruction sensitivity and early stopping along one shared training schedule; constraint enforcement, independent-seed variability, and broader scheduling generalization remain open."
    )
    words = len(re.findall(r"\b[\w'-]+\b", abstract))
    require(150 <= words <= 250, f"Measured journal abstract has {words} words; expected150--250")
    intro = (
        "The job shop scheduling problem (JSSP) assigns start times to ordered operations competing for machine capacity, commonly to minimize makespan. "
        "Classical optimization, local search, dispatching, and specialized neural policies address this problem \\cite{Beck2010JobShop,Nowicki2005TabuSearch,performance_of_HeuristicDispatchingRules,L2D,SelfJSP}. "
        "Direct text generation must satisfy every operation, precedence, and machine constraint; fluent output alone is insufficient. "
        "We investigate supervised low-rank adaptation (LoRA) \\cite{hu2022lora} of the Phi-3-Mini-128K-Instruct large language model (LLM) \\cite{abdin2024phi3} with independent post-hoc validation, without repair or constrained decoding.\n\n"
        "Our public dataset contains 120,000 unique training-source instances and 1,800 distinct released tests. The recorded revision split uses 118,000 training and 2,000 validation instances; prompt selection uses a frozen 72-instance subset of those validation records, two instances from each of the 36 size groups. "
        "Five checkpoints along one 1,844-update schedule are tested on 216 frozen instances with ten candidates each. "
        + first + "\n\n"
        "The final model is evaluated with three training instructions and four unseen wordings. A validation-only selector chooses "
        + r"\texttt{" + _escape(fixed_id) + "}. " + fixed
        + " A separate two-round proposal/refinement procedure chooses " + r"\texttt{" + _escape(auto_id) + "} before a fresh test. " + generated
        + " Neither selector uses test outcomes. The generated-instruction procedure is distinct from fixed-registry selection.\n\n"
        "Section~\\ref{sec:journal_results} retains the original comparison table and boxplot as previously reported historical results, "
        "whose original raw comparison records were not recovered and are not revalidated here. "
        "It separately reports verified training-budget, prompt, and supplementary baseline measurements. "
        "Feasibility retains all 216 instances; conditional reference gaps use the best valid candidate among ten and must be read alongside feasibility. "
        "Reference labels lack optimality certificates. Matched baseline candidate counts do not imply equal computation or training resources, and paired gap comparisons remain conditional on the stated shared subset."
    )
    retained = ("The refinement selector retained the original instruction. " if auto_id == "auto_0" else
                "The refinement selector chose a generated instruction using validation outcomes alone. ")
    conclusion = (
        "The completed single-seed study evaluates direct natural-language JSSP scheduling under fixed candidate budgets and strict validation. "
        + first + " " + fixed + " " + generated + " " + retained + "\n\n"
        "The measured checkpoint trajectory describes early stopping along one full-run learning-rate schedule; it does not compare independently optimized short-run recipes. "
        "Instruction generation and validation-guided refinement are limited to the declared two-round search. Conditional reference gaps can cover different feasible populations and do not alone establish overall superiority. "
        "The specialized-policy supplement retains full-population feasibility and paired gaps on identical Phi-feasible subsets. "
        "No result establishes a worst-case guarantee, token-by-token feasibility enforcement, performance across independent training seeds, or transfer to larger and alternative scheduling variants."
    )
    return {"journal_abstract.tex": abstract + "\n", "journal_intro.tex": intro + "\n",
            "journal_conclusion.tex": conclusion + "\n", "journal_results.tex": transform_results(primary_tex, automatic_tex, precision),
            "journal_status.tex": ""}


def transform_rebuttal(text, precision):
    """Align the frozen response rendering with the corrected journal account."""
    substitutions = {
        "Other retained diagnostics use separate populations and are not evidence from the fresh controlled run.":
            "The original comparison table and boxplot are retained as previously reported historical results. Their original raw comparison records were not recovered, so they are not revalidated here or combined with the fresh controlled run. Other earlier diagnostics are not reproduced as verified journal results.",
        "The earlier Phi-3/SLJ maximum-gap values are retained as historical reported statistics; the underlying comparison records were not recovered and are not revalidated here.":
            "The original Phi-3/SLJ comparison table and boxplot remain visible as previously reported historical statistics; the underlying comparison records were not recovered and are not revalidated here.",
        "The retained historical model-family, representation, 8B proxy, and discard statistics use their own populations; their denominators and trajectories are not combined with the new run.":
            "The earlier model-family, representation, 8B proxy, and discard diagnostics are not reproduced as verified results; their populations and trajectories are not combined with the new run.",
        "The earlier 1.2B/3.8B model-family and 8B representation comparisons are historical diagnostics, not controlled evidence about parameter count alone; their confounding factors remain stated.":
            "The earlier model-family and 8B representation diagnostics did not isolate parameter count or representation as a causal factor and are not reproduced as verified journal results.",
        "The earlier counts of 27,960 generated and 1,750 valid candidates describe the historical 577-instance experiment and have not been revalidated here.":
            "Raw records underlying the earlier 577-instance candidate-discard claims were not recovered; those counts are not reproduced as verified results.",
    }
    for old, new in substitutions.items():
        text = text.replace(old, new)
    require(PRECISION_MARKER not in text and text.count(r"\end{document}") == 1,
            "Unexpected existing rebuttal precision amendment or document boundary")
    require(not re.search(r"(?:remain(?:s)? pending|still in progress|results? (?:are|is) pending|stud(?:y|ies) (?:is|are) in progress)", text, re.I),
            "A completed response still contains pending-experiment language")
    paragraph = precision["publication_correction"]["reviewer_response_paragraph"]
    appendix = (PRECISION_MARKER + "\n" + r"\section*{Inference precision clarification}" + "\n" + paragraph
                + "\n% END VERIFIED JOURNAL PRECISION CLARIFICATION\n\n")
    return text.replace(r"\end{document}", appendix + r"\end{document}")


def integrate_verified_results(paper_dir, study_dir, runs_dir, auto_protocol_path, publisher_receipt_path):
    verified = verify_publication_chain(paper_dir, study_dir, runs_dir, auto_protocol_path, publisher_receipt_path)
    paper = verified["paper"]
    if verified["prior"]:
        return verified["prior"]
    content = render_fragments(verified["primary"], verified["automatic"],
                              (paper / "revision_results.tex").read_text(),
                              (paper / "auto_prompt_results.tex").read_text(), verified["precision"])
    content["rebuttal.tex"] = transform_rebuttal((paper / "rebuttal.tex").read_text(), verified["precision"])
    # Recheck all evidence before the first backup or journal output is created.
    for path, digest in verified["bindings"].items():
        require(file_hash(path) == digest, f"Evidence changed during journal rendering: {path}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_dir = paper / "build/journal_result_bridge" / timestamp
    backup_dir.mkdir(parents=True)
    backups = {}
    for name in (*TARGETS, "rebuttal.tex", "neurips_2024.tex", "revision_results.tex", "auto_prompt_results.tex", "neurips_2024.pdf", "rebuttal.pdf"):
        path = paper / name
        if path.exists():
            shutil.copy2(path, backup_dir / name)
            backups[name] = str(backup_dir / name)
    for name, text in content.items():
        temporary = paper / (name + ".journal_bridge.tmp")
        temporary.write_text(text)
        temporary.replace(paper / name)
    result = dict(status="verified_journal_results_integrated_pdf_rebuild_required",
                  purpose="DERIVATIVE_JOURNAL_PRESENTATION_NOT_NEW_SCIENTIFIC_RESULTS",
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  publisher_receipt_file_sha256=file_hash(publisher_receipt_path),
                  scientific_receipt_sha256=verified["published"]["scientific_receipt_sha256"],
                  integrated_receipt_sha256=verified["published"]["integrated_receipt_sha256"],
                  precision_clarification_sha256=verified["precision"]["clarification_sha256"],
                  source_code_sha256=file_hash(__file__),
                  table_reflow_code_sha256=file_hash(Path(__file__).with_name("journal_tables.py")), backups=backups,
                  input_file_sha256=verified["bindings"],
                  outputs_sha256={name: file_hash(paper / name) for name in content},
                  final_pdf_superseding_receipt_required=True)
    result["receipt_sha256"] = json_hash(result)
    receipt_path = paper / "build" / RECEIPT
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(receipt_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrate-only", action="store_true", required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--auto-protocol", type=Path, required=True)
    parser.add_argument("--publisher-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(integrate_verified_results(args.paper_dir, args.study_dir, args.runs_dir,
                                                args.auto_protocol, args.publisher_receipt), indent=2))


if __name__ == "__main__":
    main()
