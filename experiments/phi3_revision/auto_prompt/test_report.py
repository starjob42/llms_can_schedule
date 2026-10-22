"""Invented formatting/trace fixtures only; never experimental candidate evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from . import report
from . import protocol
from . import optimizer
from .test_optimizer import WordTokenizer
from core import EvaluationRecord, ValidationResult, aggregate_results


def metrics(successes,gap=0):
    good=ValidationResult(True,(),(),(),7,7,gap)
    bad=ValidationResult(False,("missing_operation",),(),(),None,None,None)
    records=[EvaluationRecord(f"fixture_{i}",candidate,good if i<successes else bad)
             for i in range(216) for candidate in range(10)]
    return aggregate_results(records,ks=(1,3,5,10))


def rendering_fixture():
    texts=[protocol.PROMPTS[0].text,
           "Give operation timings & makespan; use A_B with 50% #1 $x$ {y} \\ ~ ^.",
           "Provide the complete feasible operation schedule and final makespan.",
           "Respect every job and machine constraint and show all operation timings.",
           "Minimize the makespan and show each operation with all start and end times."]
    specs=[]
    for order,text in enumerate(texts):
        origin={"round":0,"slot":0,"attempt":0} if order==0 else {"round":(order-1)//2+1,"slot":(order-1)%2+1,"attempt":1}
        specs.append(protocol.instruction_spec(protocol.INSTRUCTION_IDS[order],text,origin,len(text.split())))
    deltas={"seen_1":{"difference_percentage_points":-25.0,"ci95_percentile":[-30,-20]},
            "fixed_registry_selected":{"difference_percentage_points":-37.5,"ci95_percentile":[-42,-32]}}
    return {"protocol":{"protocol_sha256":"SYNTHETIC_ONLY"},"complete":{"receipt_sha256":"SYNTHETIC_ONLY"},
            "instructions":specs,"selection":{"selected_instruction":specs[-1]},
            "validation_scores":{spec["id"]:{"feasible_candidates":20+i,"conditional_mean_gap_percent":1.5} for i,spec in enumerate(specs)},
            "test":{"summary":{"by_prompt":{specs[-1]["id"]:metrics(54,-2.5)}}},
            "parent_summary":{"by_prompt":{"seen_1":metrics(108),"unseen_2":metrics(135)}},
            "parent_selected_prompt":"unseen_2","comparisons":deltas}


class RenderingTests(unittest.TestCase):
    def test_tex_escape_handles_every_special_character_without_recursive_replacement(self):
        raw="%&_#$ {}\\~^"
        self.assertEqual(report.tex_escape(raw),r"\%\&\_\#\$ \{\}\textbackslash{}\textasciitilde{}\textasciicircum{}")

    def test_render_retains_negative_outcomes_budgets_and_exact_escaped_instructions(self):
        fixture=rendering_fixture()
        raw=fixture["instructions"][1]["text"]
        result=report.render(fixture)
        text=result["tex"]
        for fragment in ["54/216","-2.50", "-25.00 percentage points", "-37.50 percentage points",
                         "1,080 additional validation candidates","2,160 test candidates",
                         "1,620 are rejected (75.00", "do not imply equal total search compute",
                         "does not evaluate DSPy, per-instance instruction adaptation"]:
            self.assertIn(fragment,text)
        self.assertIn(report.tex_escape(raw),text)
        self.assertEqual(fixture["instructions"][1]["text"],raw)
        self.assertNotIn("improv",text.lower())

    def test_zero_feasible_result_has_undefined_conditional_gap(self):
        fixture=rendering_fixture();identity=fixture["selection"]["selected_instruction"]["id"]
        fixture["test"]["summary"]["by_prompt"][identity]=metrics(0)
        rendered=report.render(fixture)
        self.assertIn("0/216",rendered["outcome"])
        self.assertIn("conditional gap is undefined",rendered["outcome"])

    def test_retained_initial_instruction_is_named_without_implying_generation_won(self):
        fixture=rendering_fixture()
        fixture["selection"]["selected_instruction"]=fixture["instructions"][0]
        fixture["test"]["summary"]["by_prompt"]["auto_0"]=metrics(54,-2.5)
        rendered=report.render(fixture)
        self.assertIn("Automatic-refinement selection",rendered["tex"])
        self.assertNotIn("Generated-instruction selection",rendered["tex"])
        self.assertIn("retained the original first training instruction",rendered["outcome"])
        self.assertTrue(rendered["summary"]["retained_original_instruction"])

    def test_publication_update_is_idempotent_and_leaves_fixed_registry_study_distinct(self):
        rendered=report.render(rendering_fixture())
        paper=r"""\begin{abstract}Measured fixed study.\end{abstract}
\section{Introduction}Fixed-registry results.
\section{Related Work}Literature.
\input{revision_results}
\section{Conclusion}The automatic method selects within seven predefined instruction strings.
\section{Limitations and Future Work}
Broader automatic prompt generation, independent training seeds, LoRA hyper-parameters, and decoding search methods remain open directions.
"""
        rebuttal=r"""\begin{document}
This establishes finite validation-only selection; open-ended instruction generation, DSPy, and adaptation to other scheduling variants were not tested.
\end{document}
"""
        changed,response=report.revise_publication(paper,rebuttal,rendered)
        self.assertEqual(changed.count(r"\input{auto_prompt_results}"),1)
        self.assertIn("The fixed-registry method selects within seven",changed)
        self.assertIn("Open-ended search beyond this budget",changed)
        self.assertIn("1,080 validation candidates",response)
        self.assertIn("2,160 test candidates",response)
        self.assertNotIn("open-ended instruction generation, DSPy",response)
        self.assertEqual(report.revise_publication(changed,response,rendered),(changed,response))


class TraceReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="auto_report_synthetic_trace_")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        base=self.root/"base";base.mkdir()
        (base/"generation_config.json").write_text(json.dumps({"eos_token_id":[32000,32007,32001]}))
        self.protocol={"output_dir":str(self.root),"base_path":str(base),"protocol_sha256":"SYNTHETIC_ONLY",
                       "base_artifacts":{"sha256":"SYNTHETIC_BASE_DIGEST"},
                       "frozen_at_utc":(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()}
        self.specs=rendering_fixture()["instructions"]
        self.scores={spec["id"]:{"instruction_id":spec["id"],"text_sha256":spec["text_sha256"],
                      "split":"validation","instance_ids_sha256":"SYNTHETIC_VALIDATION_IDS", "n_instances":72,
                      "candidate_count":3,"n_candidates":216,"feasible_candidates":20+index,
                      "conditional_mean_gap_percent":1.5,"error_counts":{"missing_operation":196-index}}
                     for index,spec in enumerate(self.specs)}
        accepted=[self.specs[0]]
        for round_index in (1,2):
            eligible=list(accepted)
            incumbent=optimizer.choose_incumbent(eligible,{spec["id"]:self.scores[spec["id"]] for spec in eligible})
            for slot in (1,2):
                target=self.specs[2*(round_index-1)+slot]
                parent=self
                class ExcludedProposer:
                    tokenizer=WordTokenizer()
                    def propose(self,messages,seed):
                        return json.dumps({"instruction":target["text"]}),{
                            "base_artifacts_sha256":"SYNTHETIC_BASE_DIGEST","unadapted":True,
                            "temperature":0.8,"top_k":50,"top_p":0.95,"max_new_tokens":256,"seed":seed,
                            "output_tokens":50,"batch_seconds":1.0,
                            "runtime":{"dtype":"float16","attention_implementation":"sdpa","quantized":False,"eos_token_ids":[32000,32007,32001]}}
                actual=optimizer.propose_instruction(self.protocol,incumbent,self.scores[incumbent["id"]],accepted,round_index,slot,ExcludedProposer())
                self.assertEqual(actual,target)
                accepted.append(actual)
            scores={spec["id"]:self.scores[spec["id"]] for spec in accepted}
            value={"schema_version":1,"protocol_sha256":"SYNTHETIC_ONLY","round":round_index,
                   "eligible_instruction_ids":[spec["id"] for spec in accepted],"incumbent_before_id":incumbent["id"],
                   "selected_instruction_id":optimizer.choose_incumbent(accepted,scores)["id"],"scores":scores}
            optimizer.write_json(self.root/"rounds"/f"round_{round_index}.json",optimizer.with_hash(value,"round_sha256"))

    def replay(self):
        report._verify_attempts(self.protocol,self.specs,self.scores,WordTokenizer())

    def test_replays_all_messages_feedback_decisions_and_round_choices(self):self.replay()

    def test_mutated_feedback_or_proposal_metadata_is_rejected_even_if_rehashed(self):
        path=self.root/"proposals/round_2_slot_1/attempt_1.json"
        row=json.loads(path.read_text());row["generator_metadata"]["unadapted"]=False
        row.pop("attempt_sha256");optimizer.write_json(path,optimizer.with_hash(row,"attempt_sha256"))
        with self.assertRaisesRegex(ValueError,"unadapted base"):
            self.replay()

    def test_omitted_first_accepted_or_extra_attempts_are_rejected(self):
        path=self.root/"proposals/round_1_slot_1/attempt_2.json"
        path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,"post-acceptance"):
            self.replay()

    def test_proposer_cannot_use_a_different_incumbent_or_feedback(self):
        scores=deepcopy(self.scores);scores["auto_0"]["feasible_candidates"]=200
        with self.assertRaisesRegex(ValueError,"context, feedback"):
            report._verify_attempts(self.protocol,self.specs,scores,WordTokenizer())


class FailClosedTests(unittest.TestCase):
    def test_incomplete_auto_study_never_reads_parent_test_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake={"output_dir":temporary}
            with patch.object(report,"load_protocol",return_value=fake),patch.object(report.primary_report,"completed_evaluation") as parent:
                with self.assertRaises(FileNotFoundError):report.verify_results(Path(temporary)/"PROTOCOL.json")
                parent.assert_not_called()

    def test_scientific_or_rendering_failure_prevents_every_publication_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/"neurips_2024.tex";source.write_text("actual fixture source unchanged\n")
            with patch.object(report,"build_comparison",side_effect=ValueError("incomplete evidence")):
                with self.assertRaisesRegex(ValueError,"incomplete evidence"):
                    report.integrate(root/"PROTOCOL.json",root)
            self.assertEqual(source.read_text(),"actual fixture source unchanged\n")
            self.assertFalse((root/"build").exists());self.assertFalse((root/"auto_prompt_results.tex").exists())
            with patch.object(report,"build_comparison",return_value=report.render(rendering_fixture())),patch.object(report,"load_protocol",return_value={"output_dir":temporary}):
                with self.assertRaisesRegex(ValueError,"Parent measured-results integration"):
                    report.integrate(root/"PROTOCOL.json",root)
            self.assertEqual(source.read_text(),"actual fixture source unchanged\n")
            self.assertFalse((root/"build").exists());self.assertFalse((root/"auto_prompt_results.tex").exists())


class FullEvidenceTests(unittest.TestCase):
    """Positive metadata lifecycle; native raw-validation is tested separately."""
    setUp=TraceReplayTests.setUp
    def prepare_complete_evidence(self):
        root=self.root
        optimizer.write_json(root/"instructions/auto_0.json",self.specs[0])
        frozen=report._timestamp(self.protocol["frozen_at_utc"])
        timestamp=lambda offset:datetime.fromtimestamp(frozen+offset,timezone.utc).isoformat()
        self.protocol.update({"parent_study":str(root/"parent/STUDY.json"),"runs_dir":str(root/"parent_runs"),
                              "parent_protocol_sha256":"SYNTHETIC_PARENT", "adapter_path":str(root/"adapter"),
                              "validation_file":str(root/"parent/prompt_validation.jsonl"),
                              "test_file":str(root/"parent/test.jsonl"),"test_instance_ids":[f"fixture_{i}" for i in range(216)],
                              "training_complete":str(root/"parent_runs/main/training_complete.json")})
        optimizer.write_json(root/"parent/STUDY.json",{"prompt_registry_sha256":"SYNTHETIC_REGISTRY"})
        (root/"parent/test.jsonl").write_text("SYNTHETIC FILE-HASH FIXTURE ONLY\n")
        (root/"parent/prompt_validation.jsonl").write_text("SYNTHETIC FILE-HASH FIXTURE ONLY\n")
        optimizer.write_json(Path(self.protocol["training_complete"]),{"finished_utc":timestamp(.5)})
        run=optimizer.with_hash({"started_at_utc":timestamp(1),"adapter_artifacts":{"sha256":"SYNTHETIC_ADAPTER"}},"run_sha256")
        self.run=run
        optimizer.write_json(root/"RUN.json",run)
        good=ValidationResult(True,(),(),(),7,7,1.5)
        bad=ValidationResult(False,("missing_operation",),(),(),None,None,None)
        self.dynamic_results={}
        receipts={}
        for i,spec in enumerate(self.specs):
            folder=root/"evaluations/validation"/spec["id"]
            for name in ("evaluation_manifest.json","candidates.jsonl","COMPLETE","instruction_manifest.json"):
                optimizer.write_json(folder/name,{"purpose":"SYNTHETIC_METADATA_FIXTURE_ONLY","file":name,"instruction":spec["id"]})
            start_offset=[2,6,8,12,14][i]
            os.utime(folder/"candidates.jsonl",(frozen+start_offset+1,frozen+start_offset+1))
            receipt={"evaluation_manifest_sha256":report.sha256_file(folder/"evaluation_manifest.json"),
                     "candidates_sha256":report.sha256_file(folder/"candidates.jsonl"),
                     "complete_sha256":report.sha256_file(folder/"COMPLETE"),
                     "instruction_manifest_sha256":report.sha256_file(folder/"instruction_manifest.json")}
            base_feedback=dict(self.scores[spec["id"]],instruction_sha256=spec["text_sha256"],
                               evaluation_configuration_sha256="SYNTHETIC_EVAL_"+spec["id"],
                               raw_candidates_sha256=receipt["candidates_sha256"],protocol_sha256="SYNTHETIC_ONLY",pilot_excluded=False)
            base_feedback.pop("text_sha256")
            val_records=[EvaluationRecord(f"validation_{j//3}",j%3,good if j<20+i else bad) for j in range(216)]
            verified={"feedback":base_feedback,"receipt":receipt,"records":val_records,
                      "instruction_manifest":{"created_at_utc":timestamp(start_offset)}}
            self.dynamic_results[spec["id"]]=verified
            self.scores[spec["id"]]=report._feedback(verified,spec)
            optimizer.write_json(root/"feedback"/f"{spec['id']}.json",self.scores[spec["id"]])
            receipts[spec["id"]]=receipt
        accepted=[self.specs[0]]
        for round_index in (1,2):
            incumbent=optimizer.choose_incumbent(accepted,{spec["id"]:self.scores[spec["id"]] for spec in accepted})
            for slot in (1,2):
                path=root/"proposals"/f"round_{round_index}_slot_{slot}"/"attempt_1.json"
                saved=json.loads(path.read_text());saved.pop("attempt_sha256")
                saved["validation_feedback"]=self.scores[incumbent["id"]]
                saved["messages"]=optimizer.build_proposer_messages(incumbent,self.scores[incumbent["id"]],round_index,slot,1)
                saved["created_at_utc"]=timestamp((4 if round_index==1 else 10)+slot-1)
                optimizer.write_json(path,optimizer.with_hash(saved,"attempt_sha256"))
                accepted.append(self.specs[2*(round_index-1)+slot])
            path=root/"rounds"/f"round_{round_index}.json"
            saved=json.loads(path.read_text());saved.pop("round_sha256")
            saved["scores"]={spec["id"]:self.scores[spec["id"]] for spec in accepted}
            saved["validation_receipts"]={spec["id"]:receipts[spec["id"]] for spec in accepted}
            optimizer.write_json(path,optimizer.with_hash(saved,"round_sha256"))
        selected=optimizer.freeze_selection(self.protocol,self.specs,self.scores,receipts)
        self.selection=selected
        spec=selected["selected_instruction"]
        test_folder=root/"evaluations/test"/spec["id"]
        test_folder.mkdir(parents=True)
        test_metric=metrics(54,-2.5)
        test_records=[EvaluationRecord(f"fixture_{i}",j,ValidationResult(i<54,() if i<54 else ("missing_operation",),(),(),7 if i<54 else None,7 if i<54 else None,-2.5 if i<54 else None))
                      for i in range(216) for j in range(10)]
        self.test_result={"feedback":{},"receipt":{key:"SYNTHETIC_TEST_"+key for key in receipts[spec["id"]]},
                          "records":test_records,"summary":{"by_prompt":{spec["id"]:test_metric}},
                          "instruction_manifest":{"created_at_utc":datetime.now(timezone.utc).isoformat(),
                                                   "selected_before_test_sha256":selected["selection_sha256"]}}
        (test_folder/"candidates.jsonl").write_text("SYNTHETIC METADATA FIXTURE ONLY\n")
        test_end=report._timestamp(self.test_result["instruction_manifest"]["created_at_utc"])+1
        os.utime(test_folder/"candidates.jsonl",(test_end,test_end))
        artifacts={str(path.relative_to(root)):report.sha256_file(path) for pattern in
                   ("proposals/*/*.json","feedback/*.json","rounds/*.json") for path in sorted(root.glob(pattern))}
        complete={"status":"complete","purpose":protocol.PURPOSE,"protocol_sha256":"SYNTHETIC_ONLY",
                  "validation_schedules":1080,"test_schedules":2160,"selection_sha256":selected["selection_sha256"],
                  "test_instruction_id":spec["id"],"test_receipt":self.test_result["receipt"],"artifact_sha256":artifacts}
        optimizer.write_json(root/"COMPLETE.json",optimizer.with_hash(complete,"receipt_sha256"))
        parent_validation=root/"parent_runs/evaluations/final_prompt_validation"
        parent_validation.mkdir(parents=True)
        (parent_validation/"candidates.jsonl").write_text("SYNTHETIC FIXTURE ONLY\n")
        parent_selection={"selected_prompt_id":"seen_1","study_protocol_sha256":"SYNTHETIC_PARENT",
                          "study_test_sha256":report.sha256_file(root/"parent/test.jsonl"),
                          "prompt_registry_sha256":"SYNTHETIC_REGISTRY","validation_configuration_sha256":"SYNTHETIC_PARENT_VAL",
                          "validation_candidates_sha256":report.sha256_file(parent_validation/"candidates.jsonl")}
        optimizer.write_json(root/"parent/selected_prompt.json",parent_selection)
        os.utime(root/"parent/selected_prompt.json",(1000000000,1000000000))
        optimizer.write_json(root/"parent_runs/evaluations/final_prompt_test/evaluation_manifest.json",{})
        self.parent_records={prompt.id:[EvaluationRecord(f"fixture_{i}",j,good if i<108 else bad) for i in range(216) for j in range(10)]
                             for prompt in report.primary_report.PROMPTS}
        parent_manifest={"base_artifacts":self.protocol["base_artifacts"],"adapter_artifacts":run["adapter_artifacts"],
                         "configuration_sha256":"SYNTHETIC_PARENT_TEST"}
        val_manifest=dict(parent_manifest,configuration_sha256="SYNTHETIC_PARENT_VAL")
        self.parent_evaluations=[(parent_manifest,{"by_prompt":{prompt:metrics(108) for prompt in self.parent_records}},self.parent_records,[]),
                                 (val_manifest,{}, {prompt.id:self.dynamic_results["auto_0"]["records"] for prompt in report.primary_report.PROMPTS},[])]

    def verify_all(self,parent):
        def dynamic_result(spec,*args,split,**kwargs):
            return self.test_result if split=="test" else self.dynamic_results[spec["id"]]
        with patch.object(report,"load_protocol",return_value=self.protocol),patch.object(report,"_verify_training"),\
                patch.object(report,"_official_tokenizer",return_value=WordTokenizer()),\
                patch.object(report.dynamic,"verify_instruction",side_effect=dynamic_result),\
                patch.object(report.primary_report,"completed_evaluation",parent):
            return report.verify_results(self.root/"PROTOCOL.json")

    def test_complete_metadata_trace_passes_before_parent_comparisons_are_read(self):
        self.prepare_complete_evidence()
        from unittest.mock import Mock
        parent=Mock(side_effect=self.parent_evaluations)
        verified=self.verify_all(parent)
        self.assertEqual(parent.call_count,2)
        self.assertEqual(verified["selection"]["selected_instruction_id"],"auto_r2_s2")
        self.assertEqual(verified["comparisons"]["seen_1"]["difference_percentage_points"],-25)

    def test_bad_selection_fails_before_any_parent_test_outcomes_are_read(self):
        self.prepare_complete_evidence()
        path=self.root/"SELECTED.json"
        value=json.loads(path.read_text());value.pop("selection_sha256")
        value["selected_instruction_id"]="auto_0"
        optimizer.write_json(path,optimizer.with_hash(value,"selection_sha256"))
        from unittest.mock import Mock
        parent=Mock(side_effect=AssertionError("Parent outcomes must not be read"))
        with self.assertRaisesRegex(ValueError,"selection evidence"):
            self.verify_all(parent)
        parent.assert_not_called()

    def test_phase_order_allows_parent_training_completed_before_protocol_freeze(self):
        self.prepare_complete_evidence()
        before=report._timestamp(self.protocol["frozen_at_utc"])-30
        optimizer.write_json(Path(self.protocol["training_complete"]),{"finished_utc":datetime.fromtimestamp(before,timezone.utc).isoformat()})
        report._verify_phase_order(self.protocol,self.run,self.specs,self.dynamic_results,self.selection,self.test_result)

    def test_phase_order_rejects_selection_before_any_validation_completed(self):
        self.prepare_complete_evidence()
        selected=deepcopy(self.selection)
        selected["selected_at_utc"]=self.dynamic_results["auto_r2_s2"]["instruction_manifest"]["created_at_utc"]
        with self.assertRaisesRegex(ValueError,"finish before frozen selection"):
            report._verify_phase_order(self.protocol,self.run,self.specs,self.dynamic_results,selected,self.test_result)

    def test_phase_order_requires_all_prior_validation_for_round_proposals(self):
        self.prepare_complete_evidence()
        raw=self.root/"evaluations/validation/auto_r1_s1/candidates.jsonl"
        after_round_two=report._timestamp(json.loads((self.root/"proposals/round_2_slot_1/attempt_1.json").read_text())["created_at_utc"])+.5
        os.utime(raw,(after_round_two,after_round_two))
        with self.assertRaisesRegex(ValueError,"predates prior validation completion"):
            report._verify_phase_order(self.protocol,self.run,self.specs,self.dynamic_results,self.selection,self.test_result)

    def test_phase_order_requires_both_round_proposals_before_new_validation(self):
        self.prepare_complete_evidence()
        changed=deepcopy(self.dynamic_results)
        changed["auto_r1_s1"]["instruction_manifest"]["created_at_utc"]=json.loads((self.root/"proposals/round_1_slot_1/attempt_1.json").read_text())["created_at_utc"]
        with self.assertRaisesRegex(ValueError,"Both proposals must finish"):
            report._verify_phase_order(self.protocol,self.run,self.specs,changed,self.selection,self.test_result)

    def test_phase_order_rejects_run_before_completed_training(self):
        self.prepare_complete_evidence()
        optimizer.write_json(Path(self.protocol["training_complete"]),{"finished_utc":self.selection["selected_at_utc"]})
        with self.assertRaisesRegex(ValueError,"full training completion"):
            report._verify_phase_order(self.protocol,self.run,self.specs,self.dynamic_results,self.selection,self.test_result)


if __name__=="__main__":unittest.main()
