"""Reporting must revalidate raw candidates and reject protocol drift."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import report
    from .core import EvaluationRecord, aggregate_results, canonical_hash, parse_matrix, validate_schedule
    from .evaluate import EvaluationConfig, build_manifest, generation_seed, load_dataset
except ImportError:
    import report
    from core import EvaluationRecord, aggregate_results, canonical_hash, parse_matrix, validate_schedule
    from evaluate import EvaluationConfig, build_manifest, generation_seed, load_dataset


MATRIX = "2 2\n0 3 1 2\n1 2 0 4\n7\n"
GOLD = """Solution:
Job 0 Operation 0 on Machine 0 : 0 + 3 -> 3
Job 1 Operation 0 on Machine 1 : 0 + 2 -> 2
Job 0 Operation 1 on Machine 1 : 3 + 2 -> 5
Job 1 Operation 1 on Machine 0 : 3 + 4 -> 7
Makespan: 7
"""
MACHINE = """Optimize schedule for 2 Jobs across 2 Machines to minimize makespan.
Machine 0 is used for the following Operations:
Job 0 Operation 0 duration 3 mins.
Job 1 Operation 1 duration 4 mins.
Machine 1 is used for the following Operations:
Job 1 Operation 0 duration 2 mins.
Job 0 Operation 1 duration 2 mins.
"""
JOB = """Optimize schedule for 2 Jobs across 2 Machines to minimize makespan.
Job 0 consists of the following Operations:
Operation 0 on Machine 0 duration 3 mins.
Operation 1 on Machine 1 duration 2 mins.
Job 1 consists of the following Operations:
Operation 0 on Machine 1 duration 2 mins.
Operation 1 on Machine 0 duration 4 mins.
"""


class CompletedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "evaluation"
        self.folder.mkdir()
        self.base = self.root / "base"
        self.base.mkdir()
        # These are file-integrity fixtures, never model-loading fixtures.
        (self.base / "config.json").write_text('{}\n')
        (self.base / "model.safetensors").write_bytes(b"fixture model bytes")
        self.data = self.root / "test.jsonl"
        self.data.write_text(json.dumps({"matrix": MATRIX, "output": GOLD,
                                        "prompt_machines_first": MACHINE,
                                        "prompt_jobs_first": JOB}) + "\n")

    def make_evaluation(self, **overrides):
        self.config = EvaluationConfig(base_path=str(self.base), data=str(self.data),
                                       output_dir=str(self.folder), candidate_count=10,
                                       **overrides)
        self.items, source_count = load_dataset(self.data)
        self.manifest = build_manifest(self.config, self.items, source_count)
        self.write_manifest()
        marker = {"status": "complete", "configuration_sha256": self.manifest["configuration_sha256"],
                  "candidate_slots": self.manifest["expected_candidate_slots"]}
        (self.folder / "COMPLETE").write_text(json.dumps(marker))
        self.rows = [{"instance_id": item.instance_id, "prompt_id": prompt,
                      "candidate_id": candidate,
                      "configuration_sha256": self.manifest["configuration_sha256"],
                      "generation_seed": generation_seed(item.instance_id, self.config.seed,
                                                         candidate // self.config.batch_size),
                      "status": "generated", "output": GOLD,
                      "validation": {"feasible": True, "gap_percent": -999}}
                     for item in self.items for prompt in self.config.prompt_ids
                     for candidate in range(self.config.candidate_count)]
        self.write_candidates()

    def write_manifest(self):
        (self.folder / "evaluation_manifest.json").write_text(json.dumps(self.manifest))

    def write_candidates(self):
        (self.folder / "candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows))

    def completed(self):
        return report.completed_evaluation(self.folder, self.data, 10, ["seen_1"])

    def test_raw_outputs_override_forged_saved_feasibility_and_gaps(self):
        self.make_evaluation()
        for item in self.rows:
            item["output"] = GOLD.replace("Job 0 Operation 1 on Machine 1 : 3 + 2 -> 5\n", "")
        self.write_candidates()
        _, summary, by_prompt, _ = self.completed()
        self.assertTrue(all(not row.validation.feasible for row in by_prompt["seen_1"]))
        best = summary["by_prompt"]["seen_1"]["best_of_k"]["10"]
        self.assertEqual(best["n_eligible_instances"], 1)
        self.assertEqual(best["instances_with_feasible_candidate"], 0)
        self.assertIsNone(best["conditional_best_gap"]["mean_gap_percent"])
        self.assertIn("--", report.table_row("fixture", summary["by_prompt"]["seen_1"]))

    def test_valid_complete_population_is_recomputed(self):
        self.make_evaluation()
        _, summary, by_prompt, raw = self.completed()
        self.assertEqual(len(raw), 10)
        self.assertEqual(len(by_prompt["seen_1"]), 10)
        best = summary["by_prompt"]["seen_1"]["best_of_k"]["10"]
        self.assertEqual(best["instances_with_feasible_candidate"], 1)
        self.assertEqual(best["conditional_best_gap"]["mean_gap_percent"], 0)
        self.assertIn("1/1", report.table_row("fixture", summary["by_prompt"]["seen_1"]))

    def test_completion_marker_cannot_hide_missing_candidates(self):
        self.make_evaluation()
        self.rows.pop()
        self.write_candidates()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.completed()

    def test_skipped_candidates_cannot_enter_completed_report(self):
        self.make_evaluation()
        self.rows[0].update(status="skipped_context", output="")
        self.write_candidates()
        with self.assertRaises(ValueError):
            self.completed()

    def test_model_artifact_mutation_is_rejected(self):
        self.make_evaluation()
        (self.base / "model.safetensors").write_bytes(b"different model bytes")
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.completed()

    def test_source_or_generation_seed_mutation_is_rejected(self):
        self.make_evaluation()
        self.rows[0]["generation_seed"] += 1
        self.write_candidates()
        with self.assertRaisesRegex(ValueError, "paired seed"):
            self.completed()
        self.data.write_text(self.data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "population changed"):
            self.completed()

    def test_pilot_and_decoding_mismatch_are_rejected(self):
        self.make_evaluation(limit=1)
        with self.assertRaisesRegex(ValueError, "pilot"):
            self.completed()

    def test_wrong_batch_size_cannot_change_paired_sampling_protocol(self):
        self.make_evaluation(batch_size=2)
        with self.assertRaises(ValueError):
            self.completed()

    def test_short_generation_cap_cannot_masquerade_as_frozen_budget(self):
        self.make_evaluation(max_new_tokens=1)
        with self.assertRaises(ValueError):
            self.completed()

    def test_wrong_instance_representation_is_rejected(self):
        self.make_evaluation(input_field="prompt_jobs_first")
        with self.assertRaises(ValueError):
            self.completed()

    def test_changed_manifest_is_rejected_even_with_matching_marker_string(self):
        self.make_evaluation()
        self.manifest["generation_budget_rule"] = "Unrecorded smaller generation budget"
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.completed()


class ReportingStatisticsTests(unittest.TestCase):
    def test_headline_handles_zero_success_and_negative_reference_gap(self):
        def summary(count,gap):
            return {"best_of_k":{"10":{"instances_with_feasible_candidate":count,
                    "n_eligible_instances":216,"conditional_best_gap":{"mean_gap_percent":gap}}}}
        zero=report.measured_outcome(summary(0,None))
        self.assertIn("0/216",zero)
        self.assertIn("conditional gap is undefined",zero)
        self.assertNotIn("mean reference-makespan gap of",zero)
        self.assertIn("-3.50",report.measured_outcome(summary(5,-3.5)))
        with self.assertRaises(ValueError):report.measured_outcome(summary(0,1))

    def test_selection_sentence_does_not_assume_a_test_benefit(self):
        text=report.measured_prompt_change({"difference_percentage_points":-4.25,
                                           "ci95_percentile":[-8.0,-0.5]})
        self.assertIn("-4.25 percentage points",text)
        self.assertNotIn("improv",text.lower())
        self.assertIn("[-8.00, -0.50]",text)

    def make_study(self, root):
        study_dir = root / "study"
        study_dir.mkdir()
        identities = {}
        # Small valid problem descriptions exercise the real study verifier;
        # no training run, generated results, or large source data are needed.
        for name, count, offset in [("test.jsonl", 216, 10),
                                    ("prompt_validation.jsonl", 72, 1000)]:
            records = []
            ids = []
            for index in range(count):
                duration = offset + index
                matrix = f"2 2\n0 {duration} 1 2\n1 2 0 4\n"
                machine = MACHINE.replace("Operation 0 duration 3", f"Operation 0 duration {duration}")
                is_test=name=="test.jsonl"
                records.append({"matrix": matrix, "prompt_machines_first": machine,
                                "source_index":index if is_test else 118000+index,
                                "source_split":"released_test" if is_test else "training_source_validation",
                                "source_sha256":"FIXTURE_TEST_SOURCE" if is_test else "FIXTURE_TRAIN_SOURCE"})
                ids.append(canonical_hash(parse_matrix(matrix)))
            (study_dir / name).write_text("".join(json.dumps(record) + "\n" for record in records))
            identities[name] = ids
        snapshot = {"training_count": 118000, "validation_count": 2000,
                    "validation_indices":list(range(118000,120000)),
                    "training_source_sha256":"FIXTURE_TRAIN_SOURCE", "test_source_sha256":"FIXTURE_TEST_SOURCE",
                    "split_algorithm": "fixed fixture partition"}
        (study_dir / "PREPARATION_SPLIT.json").write_text(json.dumps(snapshot))
        prepared_path = root / "prepared_manifest.json"
        prepared_path.write_text(json.dumps(dict(snapshot, status="prepared")))
        registry = {"prompts": [asdict(prompt) for prompt in report.PROMPTS],
                    "system_prompt": report.SYSTEM_PROMPT}
        self.study = {
            "prepared_manifest": str(prepared_path),
            "artifact_sha256": {name: report.file_hash(study_dir / name)
                                for name in ["test.jsonl", "prompt_validation.jsonl", "PREPARATION_SPLIT.json"]},
            "prompt_registry_sha256": report.json_hash(registry),
            "checkpoint_steps": [0, 461, 922, 1383, 1844],
            "test_count": 216, "prompt_validation_count": 72,
            "test_instance_ids": identities["test.jsonl"],
            "prompt_validation_instance_ids": identities["prompt_validation.jsonl"],
            "test_source_indices":list(range(216)),
            "prompt_validation_source_indices":list(range(118000,118072)),
        }
        self.write_study(study_dir)
        return study_dir

    def write_study(self, study_dir):
        self.study["protocol_sha256"] = report.json_hash({key: value for key, value in self.study.items()
                                                        if key != "protocol_sha256"})
        (study_dir / "STUDY.json").write_text(json.dumps(self.study))

    def test_frozen_population_and_training_partition_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_dir = self.make_study(root)
            report.verify_study(study_dir)
            prepared = json.loads((root / "prepared_manifest.json").read_text())
            prepared["training_count"] = 117999
            (root / "prepared_manifest.json").write_text(json.dumps(prepared))
            with self.assertRaisesRegex(ValueError, "split changed"):
                report.verify_study(study_dir)
            (study_dir / "test.jsonl").write_text((study_dir / "test.jsonl").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                report.verify_study(study_dir)

    def test_frozen_registry_is_bound_to_the_prompt_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            study_dir = self.make_study(Path(temporary))
            with patch.object(report, "SYSTEM_PROMPT", "Changed system instruction"):
                with self.assertRaisesRegex(ValueError, "registry changed"):
                    report.verify_study(study_dir)

    def test_prompt_selection_population_must_belong_to_training_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            study_dir=self.make_study(Path(temporary))
            path=study_dir/"prompt_validation.jsonl"
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["source_index"]=42  # Not a held-out training-validation index.
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            self.study["artifact_sha256"][path.name]=report.file_hash(path)
            self.study["prompt_validation_source_indices"][0]=42
            self.write_study(study_dir)
            with self.assertRaisesRegex(ValueError,"not held out"):
                report.verify_study(study_dir)

    def test_frozen_manifest_checksum_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            study_dir = self.make_study(Path(temporary))
            value = json.loads((study_dir / "STUDY.json").read_text())
            value["test_count"] = 215
            (study_dir / "STUDY.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "checksum"):
                report.verify_study(study_dir)

    def test_paired_bootstrap_uses_unique_instances_and_is_order_invariant(self):
        instance = parse_matrix(MATRIX)
        good = validate_schedule(instance, GOLD)
        bad = validate_schedule(instance, "")
        first = [EvaluationRecord(f"i{index}", candidate, bad)
                 for index in range(4) for candidate in range(3)]
        second = [EvaluationRecord(f"i{index}", candidate, good)
                  for index in range(4) for candidate in range(3)]
        result = report.paired_bootstrap(first, second, repeats=100)
        self.assertEqual(result["difference_percentage_points"], 100)
        self.assertEqual(result["ci95_percentile"], [100, 100])
        self.assertEqual(result, report.paired_bootstrap(list(reversed(first)), list(reversed(second)), repeats=100))
        with self.assertRaisesRegex(ValueError, "identical"):
            report.paired_bootstrap(first, second[:-3], repeats=100)

    def test_selection_cannot_run_after_test_candidate_file_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_dir = self.make_study(root)
            target = root / "runs/evaluations/final_prompt_test"
            target.mkdir(parents=True)
            (target / "candidates.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "before final test"):
                report.select_prompt(study_dir, root / "runs")

    def test_prompt_selection_requires_the_fully_trained_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_dir = self.make_study(root)
            wrong_checkpoint_manifest = {"config": {"adapter": str(root / "runs/main/checkpoint-922")}}
            with patch.object(report, "completed_evaluation", return_value=(wrong_checkpoint_manifest, {}, {}, [])):
                with self.assertRaisesRegex(ValueError, "fully trained"):
                    report.select_prompt(study_dir, root / "runs")
            self.assertFalse((study_dir / "selected_prompt.json").exists())

    def test_partial_training_never_changes_paper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_dir = self.make_study(root)
            main = root / "runs/main"
            main.mkdir(parents=True)
            (main / "training_complete.json").write_text(json.dumps({"global_step": 1843}))
            (main / "run_manifest.json").write_text(json.dumps({"status": "running", "config": {"pilot": False}}))
            paper = root / "paper"
            paper.mkdir()
            source = paper / "neurips_2024.tex"
            source.write_text("original manuscript")
            with self.assertRaisesRegex(ValueError, "has not completed"):
                report.integrate(study_dir, root / "runs", paper)
            self.assertEqual(source.read_text(), "original manuscript")
            self.assertFalse((paper / "revision_results.tex").exists())


def build_positive_integration_fixture(root, *, full_manuscript=False, missing_section=None,
                                       missing_evaluation=None):
    """Run real integration against invented formatting fixtures in a temp tree.

    This is intentionally callable by an external TeX compile check. It never
    writes the repository paper or the real study/results directories.
    """
    root = Path(root).resolve()
    if not str(root).startswith("/tmp/"):
        raise ValueError("Formatting fixtures must remain under /tmp")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Use an empty fixture directory")
    study_dir, runs_dir, paper_dir = root / "study", root / "runs", root / "paper"
    study_dir.mkdir()
    paper_dir.mkdir()
    (root / "FIXTURE_ONLY.txt").write_text(
        "INVENTED TEST FIXTURES FOR REPORTING/TEX VALIDATION ONLY. NOT EXPERIMENTAL RESULTS.\n")

    populations = {}
    for name, count, offset in [("test.jsonl", 216, 10),
                                ("prompt_validation.jsonl", 72, 1000)]:
        rows = []
        for index in range(count):
            duration = offset + index
            matrix = f"2 2\n0 {duration} 1 2\n1 2 0 4\n"
            machine = MACHINE.replace("Operation 0 duration 3", f"Operation 0 duration {duration}")
            rows.append({"matrix": matrix, "prompt_machines_first": machine})
        (study_dir / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        populations[name] = load_dataset(study_dir / name)[0]
    registry = {"prompts": [asdict(prompt) for prompt in report.PROMPTS],
                "system_prompt": report.SYSTEM_PROMPT}
    study = {"protocol_sha256": "INVENTED_FORMATTING_FIXTURE",
             "prompt_registry_sha256": report.json_hash(registry)}
    prepared = {"status": "prepared", "tokenization": {}}
    main = runs_dir / "main"
    main.mkdir(parents=True)
    base_path = root / "fixture_base_model"
    base_path.mkdir()
    config = {"total_steps": 1844, "milestones": [0, 461, 922, 1383, 1844],
              "precision": "nf4-fp16", "max_sequence_length": 40000,
              "lora_rank": 128, "lora_alpha": 256, "lora_dropout": 0.05,
              "micro_batch_size": 2, "gradient_accumulation_steps": 32,
              "learning_rate": 0.0002, "warmup_steps": 1, "seed": 42,
              "eval_steps": 25, "attention_implementation": "sdpa", "group_by_length": True,
              "pilot": False, "base_path": str(base_path)}
    run = {"status": "complete", "config": config,
           "lora_targets": {"selected": ["qkv_proj", "o_proj"]},
           "base_model_files": {"weight_file_sha256": {"model.safetensors": "INVENTED_WEIGHT_DIGEST"}}}
    for split, key, count in [("train", "train_data", 118000),
                              ("validation", "validation_data", 2000)]:
        path = root / ("fixture_" + split + "_tokens.jsonl")
        path.write_text('{"input_ids":[1,2,3]}\n')
        digest = report.file_hash(path)
        run[key] = {"records": count, "path": str(path), "sha256": digest,
                    "max_length":19198 if split=="train" else 19073}
        prepared["tokenization"][split] = {"sha256": digest}
    (main / "run_manifest.json").write_text(json.dumps(run))
    (main / "training_complete.json").write_text(json.dumps({"global_step": 1844}))
    for step in [0, 461, 922, 1383, 1844]:
        target = main / f"checkpoint-{step}"
        target.mkdir()
        (target / "experiment_checkpoint.json").write_text(json.dumps(
            {"global_step": step, "cumulative_training_seconds": 3600 * step / 1844}))

    def records_for(items, candidates, successful_instances):
        records = []
        for index, item in enumerate(items):
            elapsed = 0
            rows = []
            for operation in item.instance.operations:
                end = elapsed + operation.duration
                rows.append(f"Job {operation.job} Operation {operation.operation} on Machine {operation.machine} : {elapsed} + {operation.duration} -> {end}")
                elapsed = end
            good_instance = replace(item.instance, reference_makespan=elapsed)
            good = validate_schedule(good_instance, "\n".join(rows) + f"\nMakespan: {elapsed}")
            bad = validate_schedule(good_instance, "")
            assert good.feasible
            for candidate_id in range(candidates):
                records.append(EvaluationRecord(item.instance_id, candidate_id,
                                                good if index < successful_instances else bad))
        return records

    prompts = [prompt.id for prompt in report.PROMPTS]
    test_items, val_items = populations["test.jsonl"], populations["prompt_validation.jsonl"]
    validation_records = {prompt: records_for(val_items, 3, 72 if prompt == "unseen_2" else 24)
                          for prompt in prompts}
    success_counts = dict(zip(prompts, [108, 120, 132, 144, 162, 84, 96]))
    final_records = {prompt: records_for(test_items, 10, success_counts[prompt]) for prompt in prompts}
    base_artifacts = {"path": str(base_path), "sha256": "INVENTED_MODEL_DIGEST",
                      "files": [{"file": "model.safetensors", "sha256": "INVENTED_WEIGHT_DIGEST"}]}

    def evaluation_manifest(step, name):
        return {"config": {"adapter": str(main / f"checkpoint-{step}"), "base_path": str(base_path)},
                "configuration_sha256": "INVENTED_" + name, "base_artifacts": base_artifacts}

    val_manifest = evaluation_manifest(1844, "validation")
    val_path = runs_dir / "evaluations/final_prompt_validation"
    val_path.mkdir(parents=True)
    (val_path / "candidates.jsonl").write_text("INVENTED FORMATTING FIXTURE, NOT CANDIDATE EVIDENCE\n")
    selection = report.select_validation_prompt(validation_records,
                                                forbidden_test_instance_ids=[item.instance_id for item in test_items])
    selection.update({"validation_configuration_sha256": val_manifest["configuration_sha256"],
                      "study_test_sha256": report.file_hash(study_dir / "test.jsonl"),
                      "validation_candidates_sha256": report.file_hash(val_path / "candidates.jsonl"),
                      "prompt_registry_sha256": study["prompt_registry_sha256"],
                      "study_protocol_sha256": study["protocol_sha256"]})
    (study_dir / "selected_prompt.json").write_text(json.dumps(selection))
    test_path = runs_dir / "evaluations/final_prompt_test"
    test_path.mkdir()
    (test_path / "evaluation_manifest.json").write_text("{}\n")
    os.utime(study_dir / "selected_prompt.json", (1000000000, 1000000000))

    def completed(folder, data_path, candidates, requested_prompts):
        name = Path(folder).name
        if name==missing_evaluation:
            raise ValueError("Synthetic fixture evaluation is incomplete")
        if name == "final_prompt_validation":
            return val_manifest, {}, validation_records, []
        if name == "final_prompt_test":
            summary = {"n_evaluated_instances": 216,
                       "by_prompt": {prompt: aggregate_results(records, ks=(1, 3, 5, 10))
                                     for prompt, records in final_records.items()}}
            raw = [{"prompt_id": prompt, "instance_id": record.instance_id,
                    "candidate_id": record.candidate_id, "hit_max_new_tokens": False}
                   for prompt, records in final_records.items() for record in records]
            return evaluation_manifest(1844, name), summary, final_records, raw
        step = int(name.removeprefix("checkpoint_"))
        counts = {0: 0, 461: 27, 922: 54, 1383: 81}
        records = {"seen_1": records_for(test_items, 10, counts[step])}
        summary = {"n_evaluated_instances": 216,
                   "by_prompt": {"seen_1": aggregate_results(records["seen_1"], ks=(1, 3, 5, 10))}}
        return evaluation_manifest(step, name), summary, records, []

    original_dir = Path(__file__).resolve().parents[2] / "LLMs_can_Schedule_paper"
    if full_manuscript:
        original_paper = (original_dir / "neurips_2024.tex").read_text()
        original_paper=original_paper.replace("\\title{LLMs can Schedule}",
            "\\title{SYNTHETIC REPORTING FIXTURE -- NOT EXPERIMENTAL RESULTS}")
        original_rebuttal = (original_dir / "rebuttal.tex").read_text()
        original_rebuttal=original_rebuttal.replace("\\title{Point-by-Point Response to Reviews}",
            "\\title{SYNTHETIC REVIEW RESPONSE FIXTURE -- NOT EXPERIMENTAL RESULTS}")
        for name in ["neurips_2024.sty", "main.bib"]:
            shutil.copyfile(original_dir / name, paper_dir / name)
        shutil.copytree(original_dir / "imgs", paper_dir / "imgs")
    else:
        original_paper = r"""\documentclass{article}
\usepackage[preprint]{neurips_2024}
\usepackage{booktabs,graphicx,listings,amsmath}
\begin{document}
\begin{abstract}
Historical headline: 46.10\% feasibility and 8.92\% gap.
\end{abstract}
\section{Introduction}
Obsolete historical headline: 46.10\% feasibility and 8.92\% gap.
\section{Related Work}
Related literature.
\section{Dataset Generation}
Existing dataset examples.
\section{Training}
In the validation reported below, we use the first prompt for all instances to keep the comparison controlled; evaluating unseen prompt phrasings is left for future robustness testing.
\section{Evaluation}
Historical test description.
\subsection{Overview of JSSP Solution Parsing and Validation}
Existing validation details.
\subsection{Diagnostic studies}\label{subsec:ablation_analysis}
\subsubsection{Analysis of Discarded Solutions and Failure Modes}
Existing failure analysis.
\section{Conclusion}
Obsolete historical headline: 46.10\% feasibility and 8.92\% gap.
\section{Limitations}
While the diagnostic studies in Section \ref{subsec:ablation_analysis} explore associations involving model configuration, problem representation, and training progress, several aspects of the prompt design remain to be systematically analyzed. Obsolete fixture limitation.

Other limitations are retained.
\end{document}
"""
        original_rebuttal = r"""\documentclass{article}
\begin{document}
\begin{enumerate}
\item \textbf{High training cost and need for reduced-training ablation.} Obsolete proxy-only fixture response.

\item \textbf{Prompt variants and evaluation prompts.} Obsolete unseen-prompt fixture response.

\item \textbf{Heuristic prompts and lack of automatic prompt adaptation.} Obsolete adaptation fixture response.

\end{enumerate}
\end{document}
"""
    if missing_section:
        original_paper=original_paper.replace("\\section{"+missing_section+"}","\\section{Intentionally Missing Fixture Section}")
    (paper_dir / "neurips_2024.tex").write_text(original_paper)
    (paper_dir / "rebuttal.tex").write_text(original_rebuttal)

    def integrate():
        with patch.object(report, "verify_study", return_value=(study, prepared)), \
                patch.object(report, "completed_evaluation", side_effect=completed):
            return report.integrate(study_dir, runs_dir, paper_dir)

    result = integrate()
    return {"root": root, "paper_dir": paper_dir, "study_dir": study_dir, "runs_dir": runs_dir,
            "original_paper": original_paper, "original_rebuttal": original_rebuttal,
            "selection": selection, "result": result, "integrate_again": integrate}


class PositiveIntegrationTests(unittest.TestCase):
    def test_verified_baseline_supplement_is_linked_and_saved_with_results(self):
        supplement={"tex":"\\paragraph{SYNTHETIC baseline hook fixture.} Formatting only.\n",
                    "summary":{"status":"synthetic_fixture_only"}}
        with tempfile.TemporaryDirectory(prefix="phi3_baseline_hook_fixture_") as temporary:
            with patch.object(report,"supplementary_baselines",return_value=supplement) as check:
                fixture=build_positive_integration_fixture(Path(temporary),full_manuscript=True)
            self.assertEqual(check.call_args.args[-1],"unseen_2")
            self.assertIn(supplement["tex"],(fixture["paper_dir"]/"revision_results.tex").read_text())
            saved=json.loads((fixture["study_dir"]/"integrated_results.json").read_text())
            self.assertEqual(saved["supplementary_baselines"],supplement["summary"])
            self.assertEqual(saved["report_code_sha256"],report.file_hash(report.__file__))
            rebuttal=(fixture["paper_dir"]/"rebuttal.tex").read_text()
            self.assertNotIn("against baselines that were not rerun",rebuttal)
            self.assertNotIn("Any separate new baseline comparison requires",rebuttal)
            self.assertIn("same 216 frozen test instances, using ten candidates",rebuttal)
            self.assertIn("Feasibility includes all 216 instances",rebuttal)
            self.assertIn("mean, median, and observed maximum reference gaps",rebuttal)
            self.assertIn("identical Phi-3-feasible subsets",rebuttal)
            self.assertIn("Equal candidate counts do not imply equal compute",rebuttal)
            self.assertIn("not a worst-case guarantee",rebuttal)

    def test_invalid_requested_baselines_block_all_publication_writes(self):
        with tempfile.TemporaryDirectory(prefix="phi3_baseline_hook_failure_") as temporary:
            root=Path(temporary)
            with patch.object(report,"supplementary_baselines",side_effect=ValueError("baseline evidence incomplete")):
                with self.assertRaisesRegex(ValueError,"baseline evidence incomplete"):
                    build_positive_integration_fixture(root)
            self.assertFalse((root/"paper/revision_results.tex").exists())
            self.assertFalse((root/"paper/build").exists())
            self.assertFalse((root/"study/integrated_results.json").exists())
            self.assertNotIn(r"\input{revision_results}",(root/"paper/neurips_2024.tex").read_text())

    def test_full_paper_uses_new_headlines_and_scopes_historical_evidence(self):
        with tempfile.TemporaryDirectory(prefix="phi3_publication_fixture_") as temporary:
            fixture=build_positive_integration_fixture(Path(temporary),full_manuscript=True)
            directory=fixture["paper_dir"]
            paper=(directory/"neurips_2024.tex").read_text()
            abstract=re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",paper,re.S)[1]
            introduction=paper.split(r"\section{Introduction}")[1].split(r"\section{Related Work}")[0]
            conclusion=paper.split(r"\section{Conclusion}")[1].split(r"\section{Limitations and Future Work}")[0]
            for passage in (abstract,introduction,conclusion):
                self.assertIn("108/216",passage)
                self.assertIn("+25.00 percentage points",passage)
                self.assertNotIn("46.10",passage)
                self.assertNotIn("8.92",passage)
                self.assertIn("reference",passage)
            training=paper.split(r"\section{Training}")[1].split(r"\section{Evaluation}")[0]
            self.assertIn('System: "You are expert in Job Shop Scheduling Problem"',training)
            self.assertIn(r'User: " {instruction} \n{problem} "',training)
            self.assertIn("system, user, and assistant tokens",training)
            self.assertIn("EOS/padding token ID are masked",training)
            self.assertIn("rejected rather than truncated",training)
            self.assertIn("19,198",training)
            self.assertIn("242 hours",training)
            self.assertIn("historical evidence",training)
            self.assertIn("1,799 planned updates",training)
            self.assertIn("durations from 1 to 499",paper)
            self.assertIn("durations from 1 to 98",paper)
            self.assertIn("not drawn from the released test source",paper)
            self.assertIn("other retained diagnostics",paper.lower())
            self.assertIn("77-instance representation and 80-instance Taillard",paper)
            self.assertIn("count & 266 & 577",paper)
            self.assertNotIn("infer superiority over specialized neural schedulers from a new study without rerunning",paper)
            results=(directory/"revision_results.tex").read_text()
            self.assertIn("Mean reference gap",results)
            self.assertIn("Reference optimality is not certified",results)
            self.assertIn("72 instances drawn only from the 2,000 training-validation records",results)
            self.assertIn("1,512 validation candidates",results)
            self.assertIn("before the final prompt-comparison test evaluation",results)
            self.assertIn("not variability across training or decoding seeds",results)
            rebuttal=(directory/"rebuttal.tex").read_text()
            self.assertIn("108/216",rebuttal)
            self.assertNotIn("Phi-3 is competitive with some baselines",rebuttal)
            self.assertNotIn("Section 4.5",rebuttal)
            self.assertNotIn(r"\ref{tab:model_size_ablation}",rebuttal)
            self.assertNotIn("A separate verified baseline supplement evaluates",rebuttal)
            self.assertIn("against baselines that were not rerun",rebuttal)
            fixture["integrate_again"]()
            self.assertEqual((directory/"neurips_2024.tex").read_text(),paper)
            self.assertEqual((directory/"rebuttal.tex").read_text(),rebuttal)

    def test_rendering_failure_does_not_write_partial_publication_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="phi3_report_failure_fixture_") as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,"Conclusion"):
                build_positive_integration_fixture(root,missing_section="Conclusion")
            self.assertFalse((root/"paper/revision_results.tex").exists())
            self.assertFalse((root/"paper/build").exists())
            self.assertIn("Historical headline",(root/"paper/neurips_2024.tex").read_text())

    def test_missing_checkpoint_evidence_blocks_all_publication_writes(self):
        with tempfile.TemporaryDirectory(prefix="phi3_report_failure_fixture_") as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,"incomplete"):
                build_positive_integration_fixture(root,missing_evaluation="checkpoint_922")
            self.assertFalse((root/"paper/revision_results.tex").exists())
            self.assertFalse((root/"paper/build").exists())
            self.assertNotIn(r"\input{revision_results}",(root/"paper/neurips_2024.tex").read_text())

    def test_verified_results_link_once_update_rebuttal_and_preserve_originals(self):
        with tempfile.TemporaryDirectory(prefix="phi3_report_fixture_") as temporary:
            fixture = build_positive_integration_fixture(Path(temporary))
            paper = fixture["paper_dir"]
            text = (paper / "revision_results.tex").read_text()
            for step in [0, 461, 922, 1383, 1844]:
                self.assertEqual(len(re.findall(rf"(?m)^{step} \(", text)), 1)
            for prompt in report.PROMPTS:
                label = re.escape(prompt.id.replace("_", r"\_") + " (")
                self.assertEqual(len(re.findall("(?m)^" + label, text)), 1)
            self.assertEqual(fixture["selection"]["selected_prompt_id"], "unseen_2")
            self.assertIn(r"It selected \texttt{unseen\_2} before the final prompt-comparison test evaluation.", text)
            manuscript = (paper / "neurips_2024.tex").read_text()
            self.assertEqual(manuscript.count(r"\input{revision_results}"), 1)
            self.assertIn("Broader automatic prompt generation, independent training seeds", manuscript)
            self.assertNotIn("Obsolete fixture limitation", manuscript)
            self.assertNotIn("evaluating unseen prompt phrasings is left for future", manuscript)
            self.assertIn("Other limitations are retained", manuscript)
            rebuttal = (paper / "rebuttal.tex").read_text()
            self.assertNotIn("Obsolete", rebuttal)
            self.assertIn("all 118,000 training examples", rebuttal)
            self.assertIn("three training instructions and four predefined unseen phrasings", rebuttal)
            self.assertIn("automatic finite instruction search", rebuttal)
            self.assertEqual(rebuttal.count(r"\section*{Controlled Phi-3 Revision Experiments}"), 1)
            self.assertEqual((paper / "build/neurips_2024.before_experimental_integration.tex").read_text(), fixture["original_paper"])
            self.assertEqual((paper / "build/rebuttal.before_experimental_integration.tex").read_text(), fixture["original_rebuttal"])
            fixture["integrate_again"]()
            self.assertEqual((paper / "neurips_2024.tex").read_text(), manuscript)
            self.assertEqual((paper / "rebuttal.tex").read_text(), rebuttal)
            saved = json.loads((fixture["study_dir"] / "integrated_results.json").read_text())
            self.assertEqual(set(saved["checkpoint_summaries"]), {"0", "461", "922", "1383", "1844"})
            self.assertEqual(saved["prompt_summary"]["n_evaluated_instances"], 216)
            self.assertEqual(saved["selected_prompt"]["n_validation_instances"], 72)
            self.assertEqual(saved["selected_prompt"]["selected_prompt_id"], "unseen_2")
            self.assertEqual(saved["prompt_candidates_hitting_token_cap"],
                             {prompt.id: 0 for prompt in report.PROMPTS})


if __name__ == "__main__":
    unittest.main()
