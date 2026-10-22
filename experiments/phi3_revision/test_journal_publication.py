"""SYNTHETIC guard/render fixtures; never write into the real paper or runs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

try:
    from . import journal_publication as bridge
    from . import journal_build as builder
except ImportError:
    import journal_publication as bridge
    import journal_build as builder


def synthetic_summary(count, mean=None):
    return {"n_instances": 216, "n_candidates": 2160,
            "feasible_candidates": count, "instances_with_any_feasible_candidate": count,
            "best_of_k": {"10": {"instances_with_feasible_candidate": count,
                                  "n_eligible_instances": 216,
                                  "conditional_best_gap": {"mean_gap_percent": mean,
                                                           "median_gap_percent": mean}}}}


def synthetic_inputs(count=0, mean=None, auto_id="auto_0"):
    summary = synthetic_summary(count, mean)
    primary = {"purpose": "SYNTHETIC_TEST_ONLY", "checkpoint_summaries": {"1844": summary},
               "selected_prompt": {"selected_prompt_id": "seen_1"},
               "prompt_summary": {"by_prompt": {"seen_1": summary}}}
    auto = {"purpose": "SYNTHETIC_TEST_ONLY", "selected_instruction": {"id": auto_id},
            "test_summary": summary}
    precision = {"publication_correction": {
        "exact_fresh_evaluation_sentence_to_replace": "SYNTHETIC OLD PRECISION.",
        "exact_replacement": "SYNTHETIC FP16 base with FP32 LoRA.",
        "reviewer_response_paragraph": "SYNTHETIC scoped precision clarification."}}
    return primary, auto, precision


def snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


class GuardTests(unittest.TestCase):
    def test_missing_training_barrier_changes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_journal_guard_") as directory:
            base = Path(directory)
            paper = base / "paper"
            paper.mkdir()
            for name in (*bridge.TARGETS, "rebuttal.tex"):
                (paper / name).write_text("SYNTHETIC EXISTING TEXT")
            before = snapshot(base)
            with self.assertRaisesRegex(bridge.PublicationGuardError, "training_complete"):
                bridge.integrate_verified_results(paper, base / "study", base / "runs",
                                                  base / "PROTOCOL.json", base / "PUBLISH_COMPLETE.json")
            self.assertEqual(snapshot(base), before)

    def test_later_missing_auto_barrier_changes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_journal_guard_") as directory:
            base = Path(directory)
            (base / "runs/main").mkdir(parents=True)
            (base / "runs/manager").mkdir()
            (base / "paper").mkdir()
            (base / "paper/rebuttal.tex").write_text("SYNTHETIC EXISTING RESPONSE")
            (base / "runs/main/training_complete.json").write_text("{}")
            (base / "runs/manager/COMPLETE").write_text("{}")
            protocol = {"output_dir": str(base / "auto")}
            protocol["protocol_sha256"] = bridge.json_hash(protocol)
            (base / "PROTOCOL.json").write_text(json.dumps(protocol))
            before = snapshot(base)
            with self.assertRaisesRegex(bridge.PublicationGuardError, "COMPLETE.json"):
                bridge.integrate_verified_results(base / "paper", base / "study", base / "runs",
                                                  base / "PROTOCOL.json", base / "PUBLISH_COMPLETE.json")
            self.assertEqual(snapshot(base), before)

    def test_selfhash_and_content_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_receipt_") as directory:
            path = Path(directory) / "receipt.json"
            value = {"purpose": "SYNTHETIC_TEST_ONLY", "count": 216}
            value["receipt_sha256"] = bridge.json_hash(value)
            path.write_text(json.dumps(value))
            bridge.load_json(path, {}, "receipt_sha256")
            original = bridge.file_hash(path)
            value["count"] = 215
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(bridge.PublicationGuardError, "Invalid receipt_sha256"):
                bridge.load_json(path, {}, "receipt_sha256")
            with self.assertRaisesRegex(bridge.PublicationGuardError, "Bound artifact changed"):
                bridge.bound_file(path, original, {})

    def test_selection_ranking_uses_complete_validation_registry(self):
        scores = {identity: {"feasible_candidates": 0, "conditional_mean_gap_percent": None}
                  for identity in bridge.PROMPT_IDS}
        self.assertEqual(bridge._rank(scores, bridge.PROMPT_IDS), "seen_1")
        scores["unseen_1"] = {"feasible_candidates": 2, "conditional_mean_gap_percent": -5.0}
        scores["unseen_2"] = {"feasible_candidates": 2, "conditional_mean_gap_percent": 2.0}
        self.assertEqual(bridge._rank(scores, bridge.PROMPT_IDS), "unseen_1")
        del scores["seen_2"]
        with self.assertRaises(bridge.PublicationGuardError):
            bridge._rank(scores, bridge.PROMPT_IDS)


class RenderTests(unittest.TestCase):
    def test_zero_and_null_outcomes_have_bounded_abstract_without_improvement_claim(self):
        for count in (0, 1):
            primary, auto, precision = synthetic_inputs(count=count)
            result = bridge.render_fragments(primary, auto, r"\subsubsection{SYNTHETIC primary}",
                                             r"\subsubsection{SYNTHETIC automatic}", precision)
            abstract = result["journal_abstract.tex"]
            self.assertGreaterEqual(len(re.findall(r"\b[\w'-]+\b", abstract)), 150)
            self.assertLessEqual(len(re.findall(r"\b[\w'-]+\b", abstract)), 250)
            self.assertNotIn("nan", abstract.lower())
            self.assertNotIn("improves", abstract.lower())
            self.assertIn("undefined" if count == 0 else "unavailable", abstract)
            self.assertEqual(result["journal_status.tex"], "")
            self.assertTrue(result["journal_results.tex"].startswith(r"\section{Results}\label{sec:journal_results}"))
            self.assertEqual(result["journal_results.tex"].count(r"\input{journal_original_comparison}"), 1)
            self.assertLess(result["journal_results.tex"].index(r"\input{journal_original_comparison}"),
                            result["journal_results.tex"].index(r"\subsection{SYNTHETIC primary}"))
            self.assertIn("job shop scheduling problem (JSSP)", result["journal_intro.tex"])
            self.assertIn("low-rank adaptation (LoRA)", result["journal_intro.tex"])
            self.assertIn("large language model (LLM)", result["journal_intro.tex"])

    def test_negative_reference_gap_and_retained_original_are_not_erased(self):
        primary, auto, precision = synthetic_inputs(count=3, mean=-2.5)
        result = bridge.render_fragments(primary, auto, "SYNTHETIC primary", "SYNTHETIC automatic", precision)
        self.assertIn(r"-2.50\%", result["journal_abstract.tex"])
        self.assertIn("retained the original instruction", result["journal_conclusion.tex"])
        invalid = deepcopy(primary)
        invalid["checkpoint_summaries"]["1844"]["n_candidates"] = 2159
        with self.assertRaisesRegex(bridge.PublicationGuardError, "216 x 10"):
            bridge.render_fragments(invalid, auto, "SYNTHETIC", "SYNTHETIC", precision)

    def test_heading_precision_and_table_reflow_preserve_measured_cells(self):
        _, _, precision = synthetic_inputs()
        raw = (r"\subsubsection{SYNTHETIC primary}\paragraph{SYNTHETIC note.} "
               "SYNTHETIC OLD PRECISION.\n"
               r"\begin{table}[!ht]\caption{SYNTHETIC TABLE}\label{tab:phi3_training_budget}"
               r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule"
               r"Method & Feasible & CI & Mean & Median \\ \midrule"
               r"SYNTHETIC & 0/216 & [0,1] & N/A & N/A \\ \bottomrule\end{tabular}}\end{table}")
        transformed = bridge.transform_results(raw, r"\subsubsection{SYNTHETIC automatic}", precision)
        self.assertIn(r"\subsection{SYNTHETIC primary}", transformed)
        self.assertIn(r"\noindent\textbf{SYNTHETIC note.}", transformed)
        self.assertNotIn(r"\resizebox", transformed)
        self.assertNotIn("SYNTHETIC OLD PRECISION", transformed)
        self.assertIn("SYNTHETIC FP16 base with FP32 LoRA", transformed)
        payload = r"SYNTHETIC & 0/216 & [0,1] & N/A & N/A \\"
        self.assertIn(payload, transformed)
        self.assertIn(payload, raw)

    def test_final_response_retains_original_comparison_with_scope_and_rejects_pending_status(self):
        _, _, precision = synthetic_inputs()
        raw = (r"\documentclass{article}\begin{document}SYNTHETIC "
               "Other retained diagnostics use separate populations and are not evidence from the fresh controlled run. "
               "The earlier Phi-3/SLJ maximum-gap values are retained as historical reported statistics; the underlying comparison records were not recovered and are not revalidated here. "
               + r"\end{document}")
        final = bridge.transform_rebuttal(raw, precision)
        self.assertNotIn("retained diagnostics", final)
        self.assertNotIn("values are retained", final)
        self.assertIn("table and boxplot are retained as previously reported historical results", final)
        self.assertIn("not revalidated here", final)
        self.assertIn("Other earlier diagnostics are not reproduced as verified journal results", final)
        self.assertIn(bridge.PRECISION_MARKER, final)
        with self.assertRaisesRegex(bridge.PublicationGuardError, "pending-experiment"):
            bridge.transform_rebuttal(raw.replace("SYNTHETIC", "SYNTHETIC results remain pending"), precision)

    def test_final_intro_keeps_original_comparison_separate_from_verified_results(self):
        primary, auto, precision = synthetic_inputs()
        result = bridge.render_fragments(primary, auto, "SYNTHETIC primary", "SYNTHETIC automatic", precision)
        self.assertIn("original comparison table and boxplot as previously reported historical results", result["journal_intro.tex"])
        self.assertIn("original raw comparison records were not recovered", result["journal_intro.tex"])
        self.assertIn("separately reports verified", result["journal_intro.tex"])


class BuildPresentationTests(unittest.TestCase):
    def test_flat_package_retains_matched_plot_and_historical_asset(self):
        self.assertIn("journal_original_comparison.tex", builder.SOURCES)
        self.assertIn("matched_gap_boxplot.pdf", builder.SOURCES)
        self.assertIn("matched_gap_boxplot.png", builder.SOURCES)
        self.assertIn("training_budget_plot.pdf", builder.SOURCES)
        self.assertIn("historical_original_box_plot.png", builder.SOURCES)
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_matched_asset_package_") as directory:
            paper, build = Path(directory) / "paper", Path(directory) / "native"
            paper.mkdir(); build.mkdir()
            for name in builder.SOURCES:
                (paper / name).write_text("SYNTHETIC source fixture: " + name)
            figure = b"SYNTHETIC matched figure-byte fixture\x00\xff"
            vector_figure = b"SYNTHETIC selectable vector figure fixture\x00\xff"
            historical = b"SYNTHETIC historical figure-byte fixture\x00\xff"
            (paper / "matched_gap_boxplot.png").write_bytes(figure)
            (paper / "matched_gap_boxplot.pdf").write_bytes(vector_figure)
            (paper / "training_budget_plot.pdf").write_bytes(vector_figure)
            (paper / "historical_original_box_plot.png").write_bytes(historical)
            for name in ("manuscript.pdf", "manuscript.bbl"):
                (build / name).write_text("SYNTHETIC native artifact")
            result = builder.package(paper, build)
            with zipfile.ZipFile(result["path"]) as archive:
                self.assertTrue(all("/" not in name for name in archive.namelist()))
                self.assertEqual(archive.read("matched_gap_boxplot.png"), figure)
                self.assertEqual(archive.read("matched_gap_boxplot.pdf"), vector_figure)
                self.assertEqual(archive.read("training_budget_plot.pdf"), vector_figure)
                self.assertEqual(archive.read("historical_original_box_plot.png"), historical)
                self.assertEqual(archive.read("journal_original_comparison.tex"), (paper / "journal_original_comparison.tex").read_bytes())
            (paper / "matched_gap_boxplot.png").unlink()
            with self.assertRaisesRegex(ValueError, "matched_gap_boxplot.png"):
                builder.package(paper, build)

    def test_removed_ai_section_is_not_required_but_declarations_still_are(self):
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_no_ai_section_") as directory:
            paper, build = Path(directory) / "paper", Path(directory) / "native"
            paper.mkdir(); build.mkdir()
            for name in ("manuscript", "rebuttal"):
                (build / (name + ".pdf")).write_text("SYNTHETIC PDF hash fixture")
                (build / (name + ".log")).write_text("SYNTHETIC clean log")
            (paper / "journal_abstract.tex").write_text(" ".join(["SYNTHETIC"] * 180))
            source = "Statements and Declarations; Data and code availability; Competing interests; Funding."
            (paper / "manuscript.tex").write_text(source)
            fake_reader = lambda path: SimpleNamespace(pages=[SimpleNamespace(
                extract_text=lambda: (
                    "SYNTHETIC readable page Matched best-of-10 feasible reference Method "
                    "(a) Feasibility (b) Gap Training budget"
                )
            )])
            with patch.dict("sys.modules", {"pypdf": SimpleNamespace(PdfReader=fake_reader)}):
                checks = builder.inspect(paper, build)
                self.assertTrue(checks["source_headings_and_required_statements"])
                (paper / "manuscript.tex").write_text(source.replace("Funding", "SYNTHETIC omitted declaration"))
                with self.assertRaisesRegex(ValueError, "Funding"):
                    builder.inspect(paper, build)


if __name__ == "__main__":
    unittest.main()
