"""Revalidate and report the separately frozen automated prompt-refinement study.

No parent test outcomes are read until the new selection and selected-prompt
test are complete. Publication writes occur only after all scientific evidence
and every proposed source replacement have been validated in memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from . import evaluate as dynamic
from . import optimizer
from .protocol import (INSTRUCTION_IDS, PURPOSE, instruction_spec, load_protocol,
                       self_hashed, verify_instruction)
from core import aggregate_results
from evaluate import artifact_digest, atomic_json, json_hash, sha256_file
import report as primary_report


def tex_escape(value):
    """Escape each original character exactly once; never alter evaluated text."""
    replacements={"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#",
                  "_":r"\_", "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}"}
    return "".join(replacements.get(character,character) for character in value)


def _official_tokenizer(protocol):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(protocol["base_path"],local_files_only=True,trust_remote_code=False)


def _timestamp(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Evidence chronology requires explicit UTC offsets")
    return result.timestamp()


def _receipt(verified):
    return dict(verified["receipt"])


def _feedback(verified,spec):
    value=dict(verified["feedback"],**_receipt(verified),instruction_id=spec["id"],
               text_sha256=spec["text_sha256"],protocol_sha256=verified["feedback"]["protocol_sha256"])
    value["feedback_sha256"]=json_hash(value)
    return value


def _verify_training(protocol,run):
    """Bind the complete production adapter to the same base and training data."""
    parent=json.loads(Path(protocol["parent_complete"]).read_text())
    training=json.loads(Path(protocol["training_complete"]).read_text())
    manifest_path=Path(protocol["runs_dir"])/"main/run_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    marker=json.loads((Path(protocol["adapter_path"])/"experiment_checkpoint.json").read_text())
    if (parent.get("status")!="complete" or parent.get("study_protocol_sha256")!=protocol["parent_protocol_sha256"]
            or training.get("global_step")!=1844 or manifest.get("status")!="complete"
            or manifest.get("config",{}).get("pilot") or marker.get("global_step")!=1844):
        raise ValueError("Automatic refinement requires the completed parent study and full non-pilot training")
    expected={"total_steps":1844,"milestones":[0,461,922,1383,1844],"precision":"nf4-fp16",
              "max_sequence_length":40000,"lora_rank":128,"lora_alpha":256,"lora_dropout":0.05,
              "micro_batch_size":2,"gradient_accumulation_steps":32,"learning_rate":0.0002,
              "warmup_steps":1,"seed":42,"eval_steps":25,"attention_implementation":"sdpa","group_by_length":True}
    if any(manifest["config"].get(key)!=value for key,value in expected.items()):
        raise ValueError("Automatic refinement's parent training recipe changed")
    if (manifest["train_data"]["records"],manifest["validation_data"]["records"])!=(118000,2000):
        raise ValueError("Automatic refinement's parent training population changed")
    if manifest["training_code_sha256"]!=protocol["source_sha256"][str(PARENT/"train.py")]:
        raise ValueError("Parent training was produced by different source code")
    if set(manifest["lora_targets"]["selected"])!={"qkv_proj","o_proj"}:
        raise ValueError("Parent actual LoRA targets changed")
    for key in ("train_data","validation_data"):
        data=manifest[key]
        if protocol["data_sha256"].get(str(Path(data["path"]).resolve()))!=data["sha256"]:
            raise ValueError("Parent training source/tokenization differs from the frozen refinement protocol")
    trained=manifest["base_model_files"]
    evaluated={row["file"]:row["sha256"] for row in protocol["base_artifacts"]["files"]}
    weights=trained.get("weight_file_sha256",{})
    if not weights or any(evaluated.get(name)!=digest for name,digest in weights.items()):
        raise ValueError("Training and refinement/proposer base-weight hashes differ")
    for name in ("config.json","tokenizer_config.json","model.safetensors.index.json"):
        if name in trained and evaluated.get(name)!=trained[name]["sha256"]:
            raise ValueError("Training and refinement base configuration/tokenizer hashes differ")
    expected_run={"purpose":PURPOSE,"protocol_sha256":protocol["protocol_sha256"],
                  "parent_protocol_sha256":protocol["parent_protocol_sha256"],
                  "adapter_artifacts":artifact_digest(Path(protocol["adapter_path"]),adapter=True),
                  "primary_manager_complete_sha256":sha256_file(Path(protocol["parent_complete"])),
                  "training_complete_sha256":sha256_file(Path(protocol["training_complete"])),
                  "training_manifest_sha256":sha256_file(manifest_path)}
    if any(run.get(key)!=value for key,value in expected_run.items()):
        raise ValueError("Refinement RUN evidence differs from the completed parent model")
    if Path(training["final_checkpoint"]).resolve()!=Path(protocol["adapter_path"]).resolve():
        raise ValueError("Training completion points to another adapter")
    return manifest


def _verify_attempts(protocol,instructions,scores,tokenizer):
    """Replay every format decision, proposal message and validation incumbent."""
    output=Path(protocol["output_dir"])
    accepted=[instructions[0]]
    expected_files=set()
    generation_stops=json.loads((Path(protocol["base_path"])/"generation_config.json").read_text())["eos_token_id"]
    generation_stops=generation_stops if isinstance(generation_stops,list) else [generation_stops]
    for round_index in (1,2):
        eligible=list(accepted)
        incumbent=optimizer.choose_incumbent(eligible,{spec["id"]:scores[spec["id"]] for spec in eligible})
        feedback=scores[incumbent["id"]]
        for slot_index in (1,2):
            spec=instructions[2*(round_index-1)+slot_index]
            previous=None
            for attempt_index in range(1,spec["origin"]["attempt"]+1):
                path=output/"proposals"/f"round_{round_index}_slot_{slot_index}"/f"attempt_{attempt_index}.json"
                expected_files.add(path)
                saved=self_hashed(path,"attempt_sha256")
                seed=optimizer.proposal_seed(round_index,slot_index,attempt_index)
                expected={"schema_version":1,"protocol_sha256":protocol["protocol_sha256"],
                          "round":round_index,"slot":slot_index,"attempt":attempt_index,"seed":seed,
                          "messages":optimizer.build_proposer_messages(incumbent,feedback,round_index,slot_index,attempt_index,previous),
                          "incumbent_instruction_id":incumbent["id"],"incumbent_text_sha256":incumbent["text_sha256"],
                          "validation_feedback":feedback,"previous_rejection":previous,
                          "accepted_instruction_ids_before_attempt":[item["id"] for item in accepted]}
                if any(saved.get(key)!=value for key,value in expected.items()):
                    raise ValueError("Proposal context, feedback, seed or incumbent differs from its frozen validation-only trace")
                if _timestamp(saved["created_at_utc"])<_timestamp(protocol["frozen_at_utc"]):
                    raise ValueError("A scientific proposal predates protocol freeze")
                metadata=saved["generator_metadata"]
                generator={"base_artifacts_sha256":protocol["base_artifacts"]["sha256"],"unadapted":True,
                           "temperature":0.8,"top_k":50,"top_p":0.95,"max_new_tokens":256,"seed":seed}
                if any(metadata.get(key)!=value for key,value in generator.items()):
                    raise ValueError("Proposal generator differs from the pinned unadapted base/configuration")
                runtime=metadata["runtime"]
                if (runtime.get("dtype")!="float16" or runtime.get("attention_implementation")!="sdpa"
                        or runtime.get("quantized") is not False or runtime.get("eos_token_ids")!=generation_stops):
                    raise ValueError("Proposal runtime precision, attention or EOS rules differ")
                if type(metadata.get("output_tokens")) is not int or not 0<=metadata["output_tokens"]<=256:
                    raise ValueError("Proposal output exceeded its fixed token budget")
                if not isinstance(metadata.get("batch_seconds"),(int,float)) or not math.isfinite(metadata["batch_seconds"]) or metadata["batch_seconds"]<0:
                    raise ValueError("Proposal timing evidence is invalid")
                try:
                    parsed=optimizer.parse_proposal(saved["raw_output"],tokenizer,[item["text"] for item in accepted])
                    reconstructed=instruction_spec(spec["id"],parsed["text"],
                                                   {"round":round_index,"slot":slot_index,"attempt":attempt_index},parsed["token_count"])
                    decision={"accepted":True,"rejection_code":None,"instruction":reconstructed}
                except optimizer.ProposalRejected as exc:
                    decision={"accepted":False,"rejection_code":exc.code,"instruction":None}
                if any(saved.get(key)!=value for key,value in decision.items()):
                    raise ValueError("Saved proposal acceptance/rejection differs from independent replay")
                if decision["accepted"]:
                    if attempt_index!=spec["origin"]["attempt"] or reconstructed!=spec:
                        raise ValueError("First accepted proposal was skipped or its exact text changed")
                elif attempt_index==spec["origin"]["attempt"]:
                    raise ValueError("Selected proposal is not format-valid")
                previous=decision["rejection_code"]
            accepted.append(spec)
        round_path=output/"rounds"/f"round_{round_index}.json"
        round_saved=self_hashed(round_path,"round_sha256")
        round_scores={item["id"]:scores[item["id"]] for item in accepted}
        expected_round={"schema_version":1,"protocol_sha256":protocol["protocol_sha256"],"round":round_index,
                        "eligible_instruction_ids":[item["id"] for item in accepted],"incumbent_before_id":incumbent["id"],
                        "selected_instruction_id":optimizer.choose_incumbent(accepted,round_scores)["id"],
                        "scores":round_scores}
        if any(round_saved.get(key)!=value for key,value in expected_round.items()):
            raise ValueError("Saved round selection differs from recomputed validation-only ranking")
    if set(output.glob("proposals/*/*.json"))!=expected_files:
        raise ValueError("Extra, missing or post-acceptance proposal attempts exist")


def _verify_phase_order(protocol,run,instructions,validation,selection,test):
    """Use immutable UTC evidence and append-only raw-file completion bounds.

    Native COMPLETE files are intentionally excluded: a completed resume rewrites
    them, while its candidates.jsonl bytes and modification time remain intact.
    """
    output=Path(protocol["output_dir"])
    frozen=_timestamp(protocol["frozen_at_utc"])
    started=_timestamp(run["started_at_utc"])
    training=json.loads(Path(protocol["training_complete"]).read_text())
    trained=_timestamp(training["finished_utc"])
    selected=_timestamp(selection["selected_at_utc"])
    if started<max(frozen,trained):
        raise ValueError("Automatic RUN must follow both protocol freeze and full training completion")
    starts={};ends={}
    for spec in instructions:
        identity=spec["id"]
        starts[identity]=_timestamp(validation[identity]["instruction_manifest"]["created_at_utc"])
        ends[identity]=(output/"evaluations/validation"/identity/"candidates.jsonl").stat().st_mtime
        if not started<=starts[identity]<=ends[identity]<=selected:
            raise ValueError("Validation must begin after RUN and finish before frozen selection")
    proposal_times=[]
    for round_index in (1,2):
        prior=INSTRUCTION_IDS[:1+2*(round_index-1)]
        prior_ready=max([started]+[ends[identity] for identity in prior])
        round_times=[]
        for spec in instructions[1+2*(round_index-1):1+2*round_index]:
            for attempt in range(1,spec["origin"]["attempt"]+1):
                path=output/"proposals"/f"round_{round_index}_slot_{spec['origin']['slot']}"/f"attempt_{attempt}.json"
                timestamp=_timestamp(json.loads(path.read_text())["created_at_utc"])
                if timestamp<prior_ready or (round_times and timestamp<round_times[-1]):
                    raise ValueError("A round proposal predates prior validation completion or its earlier attempts")
                round_times.append(timestamp)
        new=INSTRUCTION_IDS[1+2*(round_index-1):1+2*round_index]
        if max(round_times)>min(starts[identity] for identity in new):
            raise ValueError("Both proposals must finish before either new instruction's validation begins")
        proposal_times.extend(round_times)
    if selected<max([frozen,started]+list(starts.values())+list(ends.values())+proposal_times):
        raise ValueError("Selected instruction was frozen before all validation/proposal evidence existed")
    test_started=_timestamp(test["instruction_manifest"]["created_at_utc"])
    test_ended=(output/"evaluations/test"/selection["selected_instruction_id"]/"candidates.jsonl").stat().st_mtime
    if not selected<=test_started<=test_ended:
        raise ValueError("The new test must begin after frozen selection and complete after its start")


def verify_results(protocol_path):
    """Read and independently reconstruct complete scientific evidence only."""
    protocol_path=Path(protocol_path).resolve()
    protocol=load_protocol(protocol_path)
    output=Path(protocol["output_dir"])
    complete=self_hashed(output/"COMPLETE.json","receipt_sha256")
    if (complete.get("status")!="complete" or complete.get("purpose")!=PURPOSE
            or complete.get("protocol_sha256")!=protocol["protocol_sha256"]
            or complete.get("validation_schedules")!=1080 or complete.get("test_schedules")!=2160):
        raise ValueError("Automated refinement is incomplete, a pilot, or has the wrong sample budget")
    selection=self_hashed(output/"SELECTED.json","selection_sha256")
    if (selection.get("status")!="frozen_before_auto_test" or selection.get("protocol_sha256")!=protocol["protocol_sha256"]
            or selection["selection_sha256"]!=complete["selection_sha256"]):
        raise ValueError("Automatic selection evidence differs from the completed test")
    run=self_hashed(output/"RUN.json","run_sha256")
    _verify_training(protocol,run)
    if _timestamp(run["started_at_utc"])<_timestamp(protocol["frozen_at_utc"]):
        raise ValueError("Automatic refinement started before protocol freeze")
    instructions=[verify_instruction(json.loads((output/"instructions"/f"{identity}.json").read_text())) for identity in INSTRUCTION_IDS]
    if [spec["id"] for spec in instructions]!=list(INSTRUCTION_IDS):
        raise ValueError("Automatic refinement lacks exactly five instructions in creation order")
    if {path.stem for path in (output/"instructions").glob("*.json")}!=set(INSTRUCTION_IDS):
        raise ValueError("Unregistered automatic instructions exist")
    tokenizer=_official_tokenizer(protocol)
    initial_tokens=len(tokenizer.encode(instructions[0]["text"],add_special_tokens=False))
    if initial_tokens!=instructions[0]["token_count"]:
        raise ValueError("Initial instruction differs from the official tokenizer")
    scores,receipts,validation={}, {}, {}
    for spec in instructions:
        folder=output/"evaluations/validation"/spec["id"]
        verified=dynamic.verify_instruction(spec,protocol["validation_file"],protocol["adapter_path"],folder,
                    protocol["base_path"],protocol_path=protocol_path,split="validation",study_dir=Path(protocol["parent_study"]).parent)
        scores[spec["id"]]=_feedback(verified,spec)
        receipts[spec["id"]]=_receipt(verified)
        validation[spec["id"]]=verified
        saved=self_hashed(output/"feedback"/f"{spec['id']}.json","feedback_sha256")
        if saved!=scores[spec["id"]] or saved.get("pilot_excluded") or saved["n_candidates"]!=216 or saved["n_instances"]!=72 or saved["candidate_count"]!=3:
            raise ValueError("Saved validation feedback differs from the full fresh raw validation population")
        frozen=_timestamp(protocol["frozen_at_utc"])
        if _timestamp(verified["instruction_manifest"]["created_at_utc"])<frozen:
            raise ValueError("Automatic validation began before protocol freeze")
    if sum(score["n_candidates"] for score in scores.values())!=1080:
        raise ValueError("Automatic refinement requires exactly 1080 fresh validation candidates")
    if {path.name for path in (output/"evaluations/validation").iterdir() if path.is_dir()}!=set(INSTRUCTION_IDS):
        raise ValueError("Extra or missing validation evaluation slots exist")
    _verify_attempts(protocol,instructions,scores,tokenizer)
    for round_index in (1,2):
        ids=INSTRUCTION_IDS[:1+2*round_index]
        saved=self_hashed(output/"rounds"/f"round_{round_index}.json","round_sha256")
        if saved.get("validation_receipts")!={key:receipts[key] for key in ids}:
            raise ValueError("Round validation receipt bindings changed")
    expected_selected=optimizer.choose_incumbent(instructions,scores)
    if (selection.get("selected_instruction")!=expected_selected or selection.get("selected_instruction_id")!=expected_selected["id"]
            or selection.get("text_sha256")!=expected_selected["text_sha256"]
            or selection.get("scores")!=scores or selection.get("validation_receipts")!=receipts):
        raise ValueError("Frozen automatic selection is not the best validation-only instruction")
    selected_id=expected_selected["id"]
    if complete.get("test_instruction_id")!=selected_id:
        raise ValueError("Completed test used another instruction")
    test_folder=output/"evaluations/test"/selected_id
    if {path.name for path in (output/"evaluations/test").iterdir() if path.is_dir()}!={selected_id}:
        raise ValueError("More than the single selected-instruction test was evaluated")
    test=dynamic.verify_instruction(expected_selected,protocol["test_file"],protocol["adapter_path"],test_folder,
                  protocol["base_path"],protocol_path=protocol_path,split="test",study_dir=Path(protocol["parent_study"]).parent)
    if complete.get("test_receipt")!=_receipt(test):
        raise ValueError("Complete test receipt differs from independently revalidated raw evidence")
    if (_timestamp(selection["selected_at_utc"])>_timestamp(test["instruction_manifest"]["created_at_utc"])
            or test["instruction_manifest"].get("selected_before_test_sha256")!=selection["selection_sha256"]):
        raise ValueError("Automatic instruction selection was not frozen before its new test")
    _verify_phase_order(protocol,run,instructions,validation,selection,test)
    expected_artifacts={str(path.relative_to(output)):sha256_file(path) for pattern in
                        ("proposals/*/*.json","feedback/*.json","rounds/*.json") for path in sorted(output.glob(pattern))}
    if complete.get("artifact_sha256")!=expected_artifacts:
        raise ValueError("Completed proposal/feedback/round artifacts changed")
    # Only after complete auto evidence and selection chronology have passed may
    # parent test outcomes be opened for descriptive matched comparisons.
    parent_study=Path(protocol["parent_study"]).parent
    parent_selection=json.loads((parent_study/"selected_prompt.json").read_text())
    parent_selected=parent_selection["selected_prompt_id"]
    parent_manifest,parent_summary,parent_records,_=primary_report.completed_evaluation(
        Path(protocol["runs_dir"])/"evaluations/final_prompt_test",protocol["test_file"],10,
        [prompt.id for prompt in primary_report.PROMPTS])
    if (parent_manifest["base_artifacts"]!=protocol["base_artifacts"]
            or parent_manifest["adapter_artifacts"]!=run["adapter_artifacts"]
            or parent_selection["study_protocol_sha256"]!=protocol["parent_protocol_sha256"]
            or parent_selected not in parent_records):
        raise ValueError("Parent comparison does not use the same base, final adapter, population or selection")
    if (parent_selection.get("study_test_sha256")!=sha256_file(Path(protocol["test_file"]))
            or parent_selection.get("prompt_registry_sha256")!=json.loads(Path(protocol["parent_study"]).read_text())["prompt_registry_sha256"]
            or Path(parent_study/"selected_prompt.json").stat().st_mtime>(Path(protocol["runs_dir"])/"evaluations/final_prompt_test/evaluation_manifest.json").stat().st_mtime):
        raise ValueError("Parent fixed-registry selection was not frozen before its original prompt test")
    final_validation=Path(protocol["runs_dir"])/"evaluations/final_prompt_validation"
    vm,_,vr,_=primary_report.completed_evaluation(final_validation,protocol["validation_file"],3,
                                                 [prompt.id for prompt in primary_report.PROMPTS])
    recomputed=primary_report.select_validation_prompt(vr,forbidden_test_instance_ids=protocol["test_instance_ids"])
    if (recomputed["selected_prompt_id"]!=parent_selected
            or parent_selection["validation_candidates_sha256"]!=sha256_file(final_validation/"candidates.jsonl")
            or parent_selection["validation_configuration_sha256"]!=vm["configuration_sha256"]):
        raise ValueError("Parent fixed-registry selected instruction is not supported by validation evidence")
    if vm["base_artifacts"]!=protocol["base_artifacts"] or vm["adapter_artifacts"]!=run["adapter_artifacts"]:
        raise ValueError("Parent fixed-registry validation used another model")
    comparisons={"seen_1":primary_report.paired_bootstrap(parent_records["seen_1"],test["records"]),
                 "fixed_registry_selected":primary_report.paired_bootstrap(parent_records[parent_selected],test["records"])}
    return {"protocol":protocol,"complete":complete,"selection":selection,"instructions":instructions,
            "validation_scores":scores,"test":test,"parent_selected_prompt":parent_selected,
            "parent_summary":parent_summary,"comparisons":comparisons}


def _change_sentence(label,delta):
    low,high=delta["ci95_percentile"]
    return (f"Compared with {label}, test feasibility changes by {delta['difference_percentage_points']:+.2f} percentage points "
            f"(paired instance-bootstrap 95\\% interval [{low:.2f}, {high:.2f}]).")


def render(verified):
    """Pure formatting of independently checked measurements, without optimism."""
    selected=verified["selection"]["selected_instruction"]
    scores=verified["validation_scores"]
    auto_summary=verified["test"]["summary"]["by_prompt"][selected["id"]]
    parent=verified["parent_summary"]["by_prompt"]
    fixed=verified["parent_selected_prompt"]
    lines=[]
    for spec in verified["instructions"]:
        score=scores[spec["id"]]
        gap=primary_report.number(score["conditional_mean_gap_percent"])
        lines.append(f"{tex_escape(spec['id'])} & {score['feasible_candidates']}/216 & {gap}"+r" \\"+"\n")
    rows=[primary_report.table_row("Original instruction",parent["seen_1"]),
          primary_report.table_row("Fixed-registry selection",parent[fixed]),
          primary_report.table_row("Automatic-refinement selection",auto_summary)]
    total=auto_summary["n_candidates"];valid=auto_summary["feasible_candidates"]
    gap=auto_summary["best_of_k"]["10"]["conditional_best_gap"]["mean_gap_percent"]
    count=auto_summary["best_of_k"]["10"]["instances_with_feasible_candidate"]
    outcome=(f"The separately validation-selected instruction yields a feasible schedule for {count}/216 instances "
             f"({100*count/216:.2f}\\%) under a fresh ten-candidate test budget")
    if gap is not None:
        if not count or not math.isfinite(gap):raise ValueError("Conditional automatic-result gap is inconsistent with feasibility")
        outcome+=f", with a conditional mean reference-makespan gap of {gap:.2f}\\%."
    elif not count:
        outcome+=". Its conditional gap is undefined because no feasible schedule was found."
    else:
        outcome+=". Its conditional reference-gap estimate is unavailable."
    if selected["id"]=="auto_0":
        outcome+=" The automatic-refinement selector retained the original first training instruction rather than selecting a generated instruction."
    change=" ".join((_change_sentence("the original instruction",verified["comparisons"]["seen_1"]),
                     _change_sentence("the fixed-registry validation-selected instruction",verified["comparisons"]["fixed_registry_selected"])))
    instructions="\n".join(r"\item \texttt{"+tex_escape(spec["id"])+"}: "+tex_escape(spec["text"]) for spec in verified["instructions"])
    text=(r"\subsubsection{Validation-guided automatic instruction generation and refinement}\label{sec:phi3_auto_prompt}"+"\n"
          "We additionally freeze a separate follow-up protocol before its scientific outcomes. The final scheduling adapter and original seven-string study remain fixed. "
          "An unadapted Phi-3-Mini-128K-Instruct proposer, pinned to the same official base revision and content hashes, starts from the first training instruction. "
          "In each of two rounds it proposes two new instructions using the previous-round incumbent and only aggregate validation feedback: valid candidate count, conditional mean candidate reference gap, and validator error counts. "
          "Both proposals in a round use the same incumbent and feedback. No instance descriptions, reference solutions, or test outcomes are included in proposer messages. "
          "After each round, all instructions generated so far compete by valid candidate count, then conditional mean candidate reference gap, then creation order.\n\n"
          "Each of the five instructions receives three fresh schedules on each of the same 72 training-validation instances, for 1,080 additional validation candidates. "
          "The proposer samples one JSON instruction at temperature 0.8, top-$k$ 50, and top-$p$ 0.95, with a 256-token generation limit. "
          "A fixed format filter accepts the first unique, single-line printable-ASCII instruction of 20--600 characters and at most 128 official-tokenizer tokens; role delimiters and surrounding whitespace are rejected. "
          "There are at most four format attempts per instruction slot, with predetermined seeds and no manual edits or performance-based proposal retries. All attempts and exact messages are retained; exhausted attempts fail the study.\n\n"
          r"\begin{table}[!ht]\centering\small\caption{Additional automated-refinement validation results. Every instruction uses 72 instances and three candidates per instance. Candidate gaps are conditional on candidate validity and are reference-label gaps, without an established optimality certificate.}\label{tab:phi3_auto_prompt_validation}"+"\n"
          r"\begin{tabular}{lrr}\toprule Instruction & Valid candidates & Mean candidate reference gap \\ \midrule"+"\n"+"".join(lines)+
          r"\bottomrule\end{tabular}\end{table}"+"\n\n"
          f"The selector freezes \\texttt{{{tex_escape(selected['id'])}}} and all five validation receipts before a new 216-instance, ten-candidate-per-instance test, adding 2,160 test candidates. "
          "The scheduling system role, machine-centric input, native assistant framing, generation seeds, official EOS stops, and operation-dependent token caps match the parent study. "
          "Only after this new test is complete do we read the parent test outcomes for the following matched descriptive comparisons; no parent test outcome guides generation or selection.\n\n"
          r"\begin{table}[!ht]\centering\small\caption{Matched test comparison of the original instruction, fixed-registry validation selection, and automated generation/refinement selection. Every row uses the same 216 instances and ten candidates; gaps use each method's best valid candidate among ten and are conditional on its own feasible subset. Intervals are 95\% Wilson intervals. Equal per-test candidate budgets do not imply equal total search compute.}\label{tab:phi3_auto_prompt}"+"\n"
          r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule Instruction method & Feasible & Feasibility [95\% CI] & Mean reference gap & Median reference gap \\ \midrule"+"\n"+"".join(rows)+
          r"\bottomrule\end{tabular}}\end{table}"+"\n\n"+outcome+" "+change+"\n\n"
          f"Of all {total:,} newly generated test candidates, {valid:,} pass strict validation and {total-valid:,} are rejected ({100*(total-valid)/total:.2f}\\%). "
          "Conditional gaps can cover different feasible subsets and do not alone establish overall superiority. The paired bootstrap uses 10,000 resamples of unique instances, conditional on one training seed and one paired decoding-seed scheme. "
          "This implements automated instruction proposal and validation-guided refinement within a fixed two-round budget; it does not evaluate DSPy, per-instance instruction adaptation, or transfer to other scheduling variants.\n\n"
          "The exact accepted instruction strings follow; their raw bytes and hashes are retained in the experiment artifacts.\n"
          r"{\raggedright\sloppy\begin{itemize}"+"\n"+instructions+"\n"+r"\end{itemize}}"+"\n")
    summary={"status":"verified_auto_prompt_results","protocol_sha256":verified["protocol"]["protocol_sha256"],
             "complete_receipt_sha256":verified["complete"]["receipt_sha256"],
             "selected_instruction":selected,"retained_original_instruction":selected["id"]=="auto_0",
             "validation_candidates":1080,"test_candidates":2160,
             "test_summary":auto_summary,"validation_scores":scores,"parent_selected_prompt":fixed,
             "paired_feasibility_comparisons":verified["comparisons"]}
    return {"tex":text,"summary":summary,"outcome":outcome,"comparison_text":change}


def build_comparison(protocol_path):
    return render(verify_results(protocol_path))


def revise_publication(paper,rebuttal,rendered):
    """Render all scoped corrections before allowing any publication write."""
    include=r"\input{auto_prompt_results}"
    if r"\input{revision_results}" not in paper:
        raise ValueError("Parent measured-results integration must finish first")
    if include not in paper:
        paper=paper.replace(r"\input{revision_results}",r"\input{revision_results}"+"\n"+include,1)
    outcome=rendered["outcome"]
    contribution=("A separate, validation-only follow-up automatically generates and refines four new instructions in two rounds using an unadapted Phi-3 proposer. "
                  +outcome+" "+rendered["comparison_text"])
    for start_marker,end_marker in ((r"\begin{abstract}",r"\end{abstract}"),
                                   (r"\section{Introduction}",r"\section{Related Work}"),
                                   (r"\section{Conclusion}",r"\section{Limitations")):
        if start_marker not in paper:
            raise ValueError("Required publication section is absent: "+start_marker)
        start=paper.index(start_marker)+len(start_marker)
        end=paper.find(end_marker,start)
        if end<0:
            raise ValueError("Required publication section boundary is absent: "+end_marker)
        marker="% BEGIN VERIFIED AUTOMATIC PROMPT FOLLOW-UP"
        end_comment="% END VERIFIED AUTOMATIC PROMPT FOLLOW-UP"
        passage=paper[start:end]
        passage=re.sub(re.escape(marker)+r".*?"+re.escape(end_comment)+r"\s*","",passage,flags=re.S)
        passage=passage.rstrip()+"\n\n"+marker+"\n"+contribution+"\n"+end_comment+"\n\n"
        paper=paper[:start]+passage+paper[end:]
    paper=paper.replace("The automatic method selects within seven predefined instruction strings",
                        "The fixed-registry method selects within seven predefined instruction strings")
    paper=paper.replace("Broader automatic prompt generation, independent training seeds, LoRA hyper-parameters, and decoding search methods remain open directions.",
                        "Automatic instruction generation and validation-guided refinement are now tested in a separate fixed two-round study. Open-ended search beyond this budget, DSPy, per-instance instruction adaptation, independent training seeds, LoRA hyper-parameters, and decoding search methods remain open directions.")
    if rebuttal is not None:
        rebuttal=rebuttal.replace("This establishes finite validation-only selection; open-ended instruction generation, DSPy, and adaptation to other scheduling variants were not tested.",
            "This first experiment establishes finite validation-only selection. A separate follow-up additionally tests automated instruction generation and validation-guided refinement; DSPy, per-instance instruction adaptation, and transfer to other scheduling variants remain untested.")
        title=r"\section*{Additional Automatic Instruction Generation and Refinement}"
        if title in rebuttal:
            start=rebuttal.index(title);end=rebuttal.find(r"\end{document}",start)
            if end<0:raise ValueError("Rebuttal document end is absent")
            rebuttal=rebuttal[:start]+rebuttal[end:]
        response=("We additionally completed a separately frozen automated instruction-generation and refinement experiment. "
                  "Starting from the original instruction, the pinned unadapted Phi-3 base proposes two new instructions in each of two rounds using only an incumbent instruction and aggregate validation feedback. "
                  "All five instructions use 72 training-validation instances and three candidates per instance, adding 1,080 validation candidates. "
                  "Selection is frozen before a fresh 216-instance test with ten candidates each, adding 2,160 test candidates. "
                  +outcome+" "+rendered["comparison_text"]+
                  " These results address automatic generation and validation-guided refinement within the declared budget; they do not evaluate DSPy, per-instance adaptation, or transfer to other scheduling variants. "
                  "Reference gaps remain conditional on feasibility, and all proposal attempts, feedback, selections, raw candidates, and hashes are retained.")
        if r"\end{document}" not in rebuttal:raise ValueError("Rebuttal document end is absent")
        rebuttal=rebuttal.replace(r"\end{document}",title+"\n"+response+"\n\n"+r"\end{document}")
    return paper,rebuttal


def integrate(protocol_path,paper_dir):
    rendered=build_comparison(protocol_path)
    protocol=load_protocol(protocol_path)
    paper_dir=Path(paper_dir)
    source=paper_dir/"neurips_2024.tex";original=source.read_text()
    rebuttal_path=paper_dir/"rebuttal.tex"
    original_rebuttal=rebuttal_path.read_text() if rebuttal_path.exists() else None
    paper,rebuttal=revise_publication(original,original_rebuttal,rendered)
    # Every scientific and rendering guard has succeeded before any write.
    backup=paper_dir/"build/neurips_2024.before_auto_prompt_integration.tex"
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():backup.write_text(original)
    if original_rebuttal is not None:
        rb=paper_dir/"build/rebuttal.before_auto_prompt_integration.tex"
        if not rb.exists():rb.write_text(original_rebuttal)
    (paper_dir/"auto_prompt_results.tex").write_text(rendered["tex"])
    source.write_text(paper)
    if rebuttal is not None:rebuttal_path.write_text(rebuttal)
    receipt=dict(rendered["summary"],report_source_sha256=sha256_file(Path(__file__)),
                 paper_sha256=sha256_file(source),results_tex_sha256=sha256_file(paper_dir/"auto_prompt_results.tex"))
    if rebuttal is not None:receipt["rebuttal_sha256"]=sha256_file(rebuttal_path)
    receipt["receipt_sha256"]=json_hash(receipt)
    atomic_json(Path(protocol["output_dir"])/"INTEGRATED.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--paper-dir",type=Path)
    args=parser.parse_args()
    result=integrate(args.protocol,args.paper_dir) if args.paper_dir else build_comparison(args.protocol)["summary"]
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
