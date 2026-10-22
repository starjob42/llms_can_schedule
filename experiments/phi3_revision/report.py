"""Revalidate completed experiments and integrate measured results into TeX.

This refuses partial, pilot, skipped, or mismatched populations. Prompt
selection is a separate operation that must precede final test generation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from core import EvaluationRecord, PROMPTS, SYSTEM_PROMPT, select_validation_prompt, validate_schedule
from evaluate import EvaluationConfig, load_dataset, load_progress, summarize, atomic_json, artifact_digest, json_hash


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def verify_study(study_dir):
    study_dir=Path(study_dir)
    study=json.loads((study_dir / "STUDY.json").read_text())
    contents={key:value for key,value in study.items() if key!="protocol_sha256"}
    if json_hash(contents)!=study["protocol_sha256"]:
        raise ValueError("Frozen study manifest checksum is invalid")
    for name,expected in study["artifact_sha256"].items():
        if file_hash(study_dir / name)!=expected:
            raise ValueError(f"Frozen study artifact changed: {name}")
    registry={"prompts":[asdict(p) for p in PROMPTS],"system_prompt":SYSTEM_PROMPT}
    if json_hash(registry)!=study["prompt_registry_sha256"]:
        raise ValueError("Frozen prompt registry changed")
    if study["checkpoint_steps"]!=[0,461,922,1383,1844] or study["test_count"]!=216 or study["prompt_validation_count"]!=72:
        raise ValueError("Unexpected frozen study populations or milestones")
    for name,key in [("test.jsonl","test_instance_ids"),("prompt_validation.jsonl","prompt_validation_instance_ids")]:
        items,_=load_dataset(study_dir / name)
        if [i.instance_id for i in items]!=study[key]:
            raise ValueError("Frozen instance identities changed")
    if set(study["test_instance_ids"]) & set(study["prompt_validation_instance_ids"]):
        raise ValueError("Prompt selection/test contamination")
    prepared=json.loads(Path(study["prepared_manifest"]).read_text())
    snapshot=json.loads((study_dir / "PREPARATION_SPLIT.json").read_text())
    if any(prepared[key]!=value for key,value in snapshot.items()):
        raise ValueError("Prepared training split changed after freeze")
    if prepared["status"]!="prepared":
        raise ValueError("Training data preparation is incomplete")
    # The prompt-selection population belongs to the 2k training-validation
    # partition, not the separately released 1,800-instance test source.
    allowed_validation=set(snapshot["validation_indices"])
    for name,indices_key,split,source_key in (
        ("test.jsonl","test_source_indices","released_test","test_source_sha256"),
        ("prompt_validation.jsonl","prompt_validation_source_indices","training_source_validation","training_source_sha256"),
    ):
        rows=[json.loads(line) for line in (study_dir/name).read_text().splitlines() if line.strip()]
        if [row.get("source_index") for row in rows]!=study[indices_key]:
            raise ValueError("Frozen population source indices changed")
        if any(row.get("source_split")!=split or row.get("source_sha256")!=snapshot[source_key] for row in rows):
            raise ValueError("Frozen population has the wrong source provenance")
        if name=="prompt_validation.jsonl" and any(row["source_index"] not in allowed_validation for row in rows):
            raise ValueError("Prompt-validation instances are not held out from training")
    return study,prepared


def completed_evaluation(folder, data_path, candidates, prompts):
    folder, data_path = Path(folder), Path(data_path)
    manifest = json.loads((folder / "evaluation_manifest.json").read_text())
    marker = json.loads((folder / "COMPLETE").read_text())
    if json_hash({k:v for k,v in manifest.items() if k!="configuration_sha256"})!=manifest["configuration_sha256"]:
        raise ValueError("Evaluation manifest checksum is invalid")
    if manifest["run_kind"] != "FULL" or marker["status"] != "complete":
        raise ValueError(f"Refusing pilot, skipped, or incomplete run: {folder}")
    if marker["configuration_sha256"] != manifest["configuration_sha256"]:
        raise ValueError("Completion marker does not match evaluation manifest")
    if manifest["source_data_sha256"] != file_hash(data_path):
        raise ValueError(f"Evaluation population changed: {folder}")
    config = EvaluationConfig(**{**manifest["config"], "output_dir": str(folder),
                                  "prompt_ids": tuple(manifest["config"]["prompt_ids"])})
    if config.candidate_count != candidates or set(config.prompt_ids) != set(prompts):
        raise ValueError(f"Unexpected candidate/prompt budget: {folder}")
    if (config.temperature, config.top_k, config.top_p, config.seed) != (1.0,50,0.95,42):
        raise ValueError("Decoding configuration differs from the frozen study")
    if config.batch_size!=(3 if candidates==3 else 5) or config.max_new_tokens is not None or config.limit is not None or config.input_field!="prompt_machines_first":
        raise ValueError("Batch size, generation length, input format, or pilot limit differs from the frozen study")
    if manifest["frozen_prompts"]!=[asdict(p) for p in PROMPTS] or manifest["system_prompt"]!=SYSTEM_PROMPT:
        raise ValueError("Evaluation prompt text differs from the frozen registry")
    if artifact_digest(Path(config.base_path)) != manifest["base_artifacts"]:
        raise ValueError("Base model artifacts changed since evaluation")
    if config.adapter and artifact_digest(Path(config.adapter),adapter=True) != manifest["adapter_artifacts"]:
        raise ValueError("Adapter artifacts changed since evaluation")
    items, source_count = load_dataset(data_path)
    if manifest["instance_ids"] != [item.instance_id for item in items]:
        raise ValueError("Evaluation instance order differs from the frozen population")
    progress = load_progress(folder / "candidates.jsonl",items,config,manifest["configuration_sha256"])
    result = summarize(progress,items,config,manifest)
    if result["status"] != "complete" or result["terminal_candidate_slots"] != result["expected_candidate_slots"]:
        raise ValueError(f"Raw candidate population is incomplete: {folder}")
    by_prompt = {prompt: [] for prompt in prompts}
    instances = {item.instance_id:item.instance for item in items}
    for (identity,prompt,candidate), row in progress.items():
        by_prompt[prompt].append(EvaluationRecord(identity,candidate,
                                  validate_schedule(instances[identity],row["output"])))
    return manifest, result, by_prompt, list(progress.values())


def select_prompt(study_dir, runs_dir):
    study_dir, runs_dir = Path(study_dir), Path(runs_dir)
    study,_=verify_study(study_dir)
    final_path = runs_dir / "evaluations/final_prompt_test"
    if (final_path / "candidates.jsonl").exists():
        raise ValueError("Prompt selection must be frozen before final test generation")
    prompts = [p.id for p in PROMPTS]
    manifest, summary, records, _ = completed_evaluation(
        runs_dir / "evaluations/final_prompt_validation",study_dir / "prompt_validation.jsonl",3,prompts)
    if Path(manifest["config"]["adapter"]).resolve()!=(runs_dir / "main/checkpoint-1844").resolve():
        raise ValueError("Prompt selection requires the fully trained checkpoint")
    items,_ = load_dataset(study_dir / "test.jsonl")
    selection = select_validation_prompt(records,forbidden_test_instance_ids=[i.instance_id for i in items])
    selection.update({"validation_configuration_sha256":manifest["configuration_sha256"],
                      "study_test_sha256":file_hash(study_dir / "test.jsonl"),
                      "validation_candidates_sha256":file_hash(runs_dir / "evaluations/final_prompt_validation/candidates.jsonl"),
                      "prompt_registry_sha256":study["prompt_registry_sha256"],
                      "study_protocol_sha256":study["protocol_sha256"]})
    output = study_dir / "selected_prompt.json"
    if output.exists() and json.loads(output.read_text()) != selection:
        raise ValueError("An existing frozen prompt selection differs; refusing replacement")
    atomic_json(output,selection)
    return selection


def success_by_instance(records):
    result = {}
    for record in records:
        result[record.instance_id] = result.get(record.instance_id,False) or record.validation.feasible
    return result


def paired_bootstrap(a,b,repeats=10000):
    """Paired instance bootstrap for feasibility difference, percentage points."""
    a,b = success_by_instance(a),success_by_instance(b)
    if not a or a.keys()!=b.keys():
        raise ValueError("Paired comparisons require identical nonempty populations")
    delta=[int(b[key])-int(a[key]) for key in sorted(a)]
    rng=random.Random(42)
    samples=sorted(100*sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(repeats))
    return {"difference_percentage_points":100*statistics.mean(delta),
            "ci95_percentile":[samples[int(.025*repeats)],samples[min(repeats-1,int(.975*repeats))]],
            "resampling_unit":"unique JSSP instance","repeats":repeats,"seed":42}


def percent(value):
    return "--" if value is None else f"{100*value:.2f}\\%"


def number(value):
    return "--" if value is None else f"{value:.2f}\\%"


def table_row(label,summary,k=10,extra=""):
    item=summary["best_of_k"][str(k)]
    lo,hi=item["instance_feasibility_ci95_wilson"]
    gap=item["conditional_best_gap"]
    return (f"{label}{extra} & {item['instances_with_feasible_candidate']}/{item['n_eligible_instances']} & "
            f"{percent(item['instance_feasibility_rate'])} [{percent(lo)}, {percent(hi)}] & "
            f"{number(gap['mean_gap_percent'])} & {number(gap['median_gap_percent'])} \\\\\n")


def supplementary_baselines(study_dir,runs_dir,records,summary,selected):
    """Include the separately frozen study only when present beside these runs.

    The baseline helper revalidates its complete raw population in the original
    legacy environment. A present but invalid/incomplete study fails closed.
    """
    baseline_dir=Path(runs_dir).parent / "baselines"
    if not (baseline_dir / "artifacts/PROTOCOL.json").is_file():
        return None
    try:
        from .baselines.report import build_comparison
    except ImportError:
        from baselines.report import build_comparison
    return build_comparison(study_dir,runs_dir,baseline_dir,records,summary,selected)


def measured_outcome(summary):
    """Describe the measured baseline without presuming feasibility or improvement."""
    item=summary["best_of_k"]["10"]
    count,total=item["instances_with_feasible_candidate"],item["n_eligible_instances"]
    mean=item["conditional_best_gap"]["mean_gap_percent"]
    if total!=216 or not 0<=count<=total:
        raise ValueError("Headline results require the complete 216-instance population")
    result=(f"The fully trained Phi-3 model produces a feasible schedule for {count}/{total} instances "
            f"({100*count/total:.2f}\\%) using exactly ten candidates per instance")
    if mean is not None:
        if not count or not math.isfinite(mean):
            raise ValueError("Conditional reference gap is inconsistent with measured feasibility")
        result+=f", with a conditional mean reference-makespan gap of {mean:.2f}\\%"
    result+="."
    if not count:
        result+=" No feasible schedule is found under this budget, so a conditional gap is undefined."
    elif mean is None:
        result+=" A conditional reference-gap estimate is unavailable."
    return result


def measured_prompt_change(delta):
    difference=delta["difference_percentage_points"]
    low,high=delta["ci95_percentile"]
    if not all(math.isfinite(value) for value in (difference,low,high)) or low>high:
        raise ValueError("Prompt feasibility interval is invalid")
    return (f"Validation-only instruction selection changes test feasibility by {difference:+.2f} percentage points "
            f"relative to the first training instruction (paired instance-bootstrap 95\\% interval "
            f"[{low:.2f}, {high:.2f}]).")


def replace_section(text,title,body):
    """Replace a named main section, retaining the next section and bibliography."""
    marker="\\section{"+title+"}"
    if text.count(marker)!=1:
        raise ValueError(f"Expected one manuscript section: {title}")
    start=text.index(marker)+len(marker)
    match=re.search(r"(?m)^\\(?:section\{|bibliographystyle\{|newpage\b|end\{document\})",text[start:])
    end=start+match.start() if match else len(text)
    return text[:start]+"\n\n"+body.rstrip()+"\n\n"+text[end:]


def prefix_historical_caption(text,label):
    marker="\\label{"+label+"}"
    if marker not in text:
        return text
    caption=text.rfind("\\caption{",0,text.index(marker))
    if caption<0:
        raise ValueError(f"Historical figure/table has no caption: {label}")
    start=caption+len("\\caption{")
    prefix="Historical results, retained separately from the fresh controlled study. "
    if not text[start:].startswith(prefix):
        text=text[:start]+prefix+text[start:]
    return text


def revise_publication(paper,baseline,delta,selected,hours,run):
    """Render current evidence and isolate unrecovered historical claims.

    Called only after integrate() verifies training, every evaluation, selection,
    and model/data hashes. This function itself performs no filesystem writes.
    """
    outcome=measured_outcome(baseline)
    prompt_change=measured_prompt_change(delta)
    historical=("The historical 577-instance Phi-3 evaluation manifest and raw candidate records were not recovered, so that comparison has not been revalidated. "
                "Other retained diagnostics use separate populations and are not evidence from the new controlled run.")
    abstract=("We study whether supervised fine-tuning enables a large language model to emit complete schedules for the classical job shop scheduling problem. "
              "We provide a public dataset of 120,000 natural-language instances paired with feasible solver schedules. "
              "A fresh single-seed Phi-3-Mini-128K-Instruct run is evaluated on a frozen stratified set of 216 unique instances. "
              +outcome+" We compare initialized, 25, 50, 75, and 100\\% training checkpoints under matched decoding budgets, and evaluate three training instruction strings and four unseen strings. "
              +prompt_change+" Reference gaps are conditional on feasibility and do not establish optimality. "
              "Historical neural-baseline comparisons use a different, unrecovered test population.")
    if len(re.findall(r"\\begin\{abstract\}.*?\\end\{abstract\}",paper,re.S))!=1:
        raise ValueError("Expected one manuscript abstract")
    paper=re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
                 lambda _:"\\begin{abstract}\n"+abstract+"\n\\end{abstract}",paper,count=1,flags=re.S)
    introduction=(
        "The job shop scheduling problem (JSSP) assigns start times to ordered job operations on capacity-limited machines, usually to minimize makespan, the maximum operation completion time. "
        "Traditional optimization and dispatching methods and specialized neural schedulers address this combinatorial problem \\cite{performance_of_HeuristicDispatchingRules,L2D,SelfJSP}.\n\n"
        "We investigate supervised LoRA fine-tuning of Phi-3-Mini-128K-Instruct for generating complete schedules in natural language. "
        "Our public dataset contains 120,000 unique instances paired with feasible solver-generated schedules. "
        "The fresh controlled revision uses 118,000 examples for training and 2,000 for validation, and evaluates a frozen, outcome-blind stratified subset of 216 released test instances. "
        +outcome+"\n\n"
        "The revision compares five checkpoints along one fixed full-run learning-rate schedule and tests seven introductory instruction strings while keeping instances and candidate budgets fixed. "
        "Automatic selection among these strings uses 72 instances from the 2,000-example training-validation partition and is frozen before the final prompt-comparison test evaluation. "
        +prompt_change+" The new results are reported in Section \\ref{sec:phi3_revision}.\n\n"
        +historical+" The checkpoint-and-prompt comparisons do not themselves establish performance relative to specialized schedulers; any separate baseline comparison requires matched populations and budgets.\n\n"
        "The contributions are the public natural-language training dataset, a controlled single-seed Phi-3 training-budget study, and matched instruction-wording and validation-only instruction-selection experiments with retained raw outputs and strict schedule validation.")
    paper=replace_section(paper,"Introduction",introduction)

    training_marker="\\section{Training}"
    training_start=paper.index(training_marker)+len(training_marker)
    training_end=paper.index("\\section{Evaluation}",training_start)
    historical_figures="\n\n".join(re.findall(r"\\begin\{figure\}.*?\\end\{figure\}",paper[training_start:training_end],re.S))
    prompt_list="\n".join("\\item \\texttt{"+prompt.text+"}" for prompt in PROMPTS if prompt.seen_in_training)
    maximum=run["train_data"].get("max_length")
    maximum_sentence=(f"The longest observed training sequence has {maximum:,} tokens. " if maximum is not None else "")
    training=(r"\label{sec:fresh_training}"+"\n"
        "This section describes the fresh controlled revision run. A recorded seed-42 shuffle partitions the 120,000 released instances into 118,000 training and 2,000 validation examples. "
        "Each example uses the machine-centric description and one randomly selected instruction from the following fixed set:\n"
        r"\begin{itemize}"+"\n"+prompt_list+"\n"+r"\end{itemize}"+"\n\n"
        "The released training chat template is preserved, including its system string and surrounding spaces:\n"
        r"\begin{lstlisting}"+"\n"
        'System: "You are expert in Job Shop Scheduling Problem"\n'
        r'User: " {instruction} \n{problem} "'+"\n"
        'Assistant: " {solution} "\n'+r"\end{lstlisting}"+"\n"
        "The tokenizer applies the native chat template without an additional generation prefix for complete training conversations. "
        "Examples are unpadded on disk and dynamically padded on the right within each micro-batch, using the tokenizer EOS token as padding. "
        "Sequences exceeding the 40,000-token cap are rejected rather than truncated. "+maximum_sentence+
        "The causal-language-model loss includes system, user, and assistant tokens; positions equal to the EOS/padding token ID are masked with $-100$ by \\texttt{DataCollatorForLanguageModeling(mlm=False)}.\n\n"
        "The base model is loaded in 4-bit NF4 with FP16 computation and no double quantization. LoRA uses rank 128, alpha 256, dropout 0.05, and no bias adaptation. "
        "The actual adapted modules are \\texttt{qkv\\_proj} and \\texttt{o\\_proj}; the legacy requested names \\texttt{fc1} and \\texttt{fc2} are absent from this architecture. "
        "Gradient checkpointing is enabled. A micro-batch of two and 32 accumulation steps give effective batch size 64 on one GPU. "
        "We use \\texttt{paged\\_adamw\\_8bit}, learning rate $2\\times10^{-4}$, one warm-up update, and one linear learning-rate schedule over 1,844 optimizer updates. "
        "This completes one epoch over all 118,000 examples, including the final partial accumulation batch. Training and validation losses are logged every 25 updates. "
        "Initialized and 25, 50, 75, and 100\\% checkpoints preserve optimizer and random-number-generator state. "
        f"The measured active run took {hours:.2f} hours, including validation and checkpoint overhead; setup and time between resumed processes are excluded. "
        "One training seed (42) is used.\n\n"
        "The historical manuscript reported approximately 242 hours on an A6000 GPU. This timing and the existing loss/gradient figures below are historical evidence, not measurements of the fresh run. "
        "The public training script specifies two epochs with a 118,000/2,000 split, whereas recovered checkpoint-1750 metadata specifies one epoch and 1,799 planned updates. "
        "Its exact split identities were not recovered; an approximately 115k/5k split is inferred, not verified. We therefore do not present the fresh run as an exact replay of that checkpoint.\n\n"
        +historical_figures)
    paper=replace_section(paper,"Training",training)

    evaluation_start=paper.index("\\section{Evaluation}")+len("\\section{Evaluation}")
    parsing_start=paper.index("\\subsection{Overview of JSSP Solution Parsing and Validation}",evaluation_start)
    evaluation=(
        "The verified revision evaluates 216 unique test instances, selected without using outcomes: six from each of the 36 size groups in the public 1,800-instance release. "
        "The released test source contains 50 instances per group, with sizes from 2x2 through 9x9 and observed operation durations from 1 to 98. "
        "The separate prompt-selection population consists of 72 instances, two per test-size group, drawn only from the 2,000 training-validation examples; it is not drawn from the released test source. "
        "Canonical input hashes establish that training, prompt selection, and final tests do not overlap.\n\n"
        "Fresh evaluations use FP16 inference without quantization, native Phi-3 with SDPA, and exactly ten candidates per instruction and instance, in batches of five. "
        "The system string is \\texttt{You are an expert in Job Shop Scheduling Problem}; user text is the instruction, one newline, and the unchanged machine-centric instance description. "
        "The tokenizer adds the assistant generation prefix and generation honors all official end-token IDs. These evaluation framing choices are fixed across all fresh comparisons and differ from the complete-conversation training wrapper. "
        "For $n$ operations the generation cap is $\\min(8192,48n+128)$ new tokens. No attempt is stopped because an earlier candidate is feasible, and no generated schedule is repaired.\n\n"
        "For a feasible instance we select the valid candidate with minimum makespan $C_{\\mathrm{best}}$ and define its reference gap as $100(C_{\\mathrm{best}}-C_{\\mathrm{ref}})/C_{\\mathrm{ref}}$, where $C_{\\mathrm{ref}}$ is the released reference makespan. "
        "No optimality certificate is established for these references; gaps can be negative. Mean and median gaps exclude infeasible instances, so their feasible subsets may differ between checkpoints or instructions. "
        "Feasibility always uses the full frozen instance denominator.\n\n"
        "The following historical baseline, hyperparameter, and proxy-model studies are retained on their own reported populations. "
        "The 577-instance Phi-3 comparison cannot be revalidated because its evaluation manifest and raw candidate records were not recovered. "
        "The separate 77-instance representation and 80-instance Taillard proxy diagnostics are not results from the fresh run. "
        "The earlier description of approximately 1,000 test records with durations 1--199 is not a description of the audited public release. "
        "All 1,800 released tests have fewer than 100 operations, so that threshold cannot reconstruct the historical 577-instance subset. "
        "The historical inference memory estimate of about 43 GB and up-to-60-candidate protocol apply only to that earlier setup. "
        "Verified new results appear in Section \\ref{sec:phi3_revision}.\n\n")
    paper=paper[:evaluation_start]+"\n\n"+evaluation+paper[parsing_start:]
    # Keep old tables/numbers intact, but scope every retained study explicitly.
    paper=paper.replace("\\subsection{Hyperparameter Tuning}","\\subsection{Historical Hyperparameter Tuning}")
    paper=paper.replace("\\subsection{Comparative Analysis with Other Neural Approaches}","\\subsection{Historical Comparison with Other Neural Approaches}")
    paper=paper.replace("\\subsubsection{Impact of Training Budget (Checkpoint Analysis)}", "\\subsubsection{Historical 8B Proxy Checkpoint Analysis}")
    paper=paper.replace("Although the primary experiments in this paper are based on the fine-tuned 3.8B parameter Phi-3 model (which took 242 hours of GPU training), intermediate checkpoints for Phi-3 were not preserved during the 10-day training run.",
                       "Intermediate checkpoints from the historical 242-hour Phi-3 run were not recovered; this historical proxy analysis is separate from the fresh controlled Phi-3 trajectory in Section \\ref{sec:phi3_revision}.")
    for label in ("fig:both_figures","fig:box_plot","tab:test_summary_stats","tab:model_size_ablation",
                  "tab:representation_ablation","tab:checkpoint_ablation","tab:failure_modes"):
        paper=prefix_historical_caption(paper,label)
    paper=re.sub(r"We have generated approximately 120,000 random JSSP problems.*?The dataset is publicly available at the URL in the footnote above\.",
                 lambda _:"The audited public release contains exactly 120,000 unique training-source instances of sizes from 2x2 to 20x20, including asymmetric shapes, with observed operation durations from 1 to 499 units \\footnote{\\url{https://github.com/starjob42/datasetjsp}}. "
                 "The fresh revision partitions these into 118,000 training and 2,000 validation records. All released labels pass the strict feasibility validator, and no canonical training-source instance overlaps the 1,800-instance released test source. "
                 "OR-Tools labels were produced under a 300-second budget; passing feasibility checks does not certify optimality, and no reference optimality certificates are established here.",paper,count=1,flags=re.S)
    conclusion=(
        "The fresh controlled revision evaluates Phi-3-Mini-128K-Instruct on 216 frozen unique instances under a matched ten-candidate budget. "+outcome+"\n\n"
        "The checkpoint table reports the measured training-budget trajectory; its conditional gap values must be read together with feasibility because the feasible subsets can change. "
        "These checkpoint comparisons measure early stopping along one shared full-run schedule, rather than independently optimized short-run schedules. "
        +prompt_change+" The automatic method selects within seven predefined instruction strings using 72 training-validation instances; it does not generate new instructions or establish transfer to other scheduling variants.\n\n"
        +historical+" We do not infer superiority over specialized neural schedulers from checkpoint-and-prompt comparisons alone. "
        "Constraint enforcement, repair, larger-instance generalization, and variability across independent training and decoding seeds remain open questions.")
    paper=replace_section(paper,"Conclusion",conclusion)
    old_limit="While the diagnostic studies in Section \\ref{subsec:ablation_analysis} explore associations involving model configuration, problem representation, and training progress, several aspects of the prompt design remain to be systematically analyzed."
    new_limit=("Section \\ref{sec:phi3_revision} adds controlled Phi-3 checkpoint and instruction-wording experiments, including validation-only automatic selection. "
               "Broader automatic prompt generation, independent training seeds, LoRA hyper-parameters, and decoding search methods remain open directions. "
               "The study has one training seed and one paired decoding-seed scheme; Wilson intervals and the instance bootstrap describe instance-level uncertainty conditional on that run, not variability across training or decoding seeds.")
    if old_limit in paper:
        start=paper.index(old_limit);end=paper.find("\n\n",start)
        if end<0:raise ValueError("Unexpected limitations paragraph structure")
        paper=paper[:start]+new_limit+paper[end:]
    return paper


def revise_rebuttal(rebuttal,baseline,delta,supplementary=None):
    outcome=measured_outcome(baseline)
    prompt_change=measured_prompt_change(delta)
    historical=("The historical 577-instance Phi-3 evaluation manifest and raw candidate records were not recovered, so that comparison has not been revalidated. "
                "Other retained diagnostics use separate populations and are not evidence from the fresh controlled run.")
    opening=("We completed a fresh controlled Phi-3 training-budget study and matched instruction-wording and validation-only instruction-selection experiments. "
             +outcome+" "+prompt_change+" "+historical+
             " The new results use 216 unique frozen tests and conditional reference-label gaps, without claiming superiority over specialized baselines.")
    if "\\maketitle" in rebuttal and "\\section*{Response to Reviewer 1}" in rebuttal:
        start=rebuttal.index("\\maketitle")+len("\\maketitle")
        end=rebuttal.index("\\section*{Response to Reviewer 1}",start)
        rebuttal=rebuttal[:start]+"\n\n"+opening+"\n\n"+rebuttal[end:]
    replacements={
        "High training cost and need for reduced-training ablation.":
            "We added a fresh controlled run using all 118,000 training examples and a separate 2,000-example validation split. The initialized, 25\\%, 50\\%, 75\\%, and 100\\% checkpoints share the same data order and full-run learning-rate schedule, and each is evaluated on the same 216 unique tests with exactly ten candidates. Measured active-run hours, feasibility intervals, and conditional reference gaps appear in the controlled Phi-3 subsection. "+outcome+" "+historical,
        "Prompt variants and evaluation prompts.":
            "We evaluate three training instructions and four predefined unseen phrasings using the final Phi-3 checkpoint on the same 216 tests with ten candidates per instruction. Automatic validation-only selection among the seven instructions uses three candidates on each of 72 instances drawn from the 2,000 training-validation records. The prompt registry, selection rule, and populations were frozen before outcomes, and the selected instruction was frozen before the final prompt-comparison test evaluation. "+prompt_change+" Exact instruction strings, chat wrappers, raw outputs, and selection evidence are retained.",
        "Heuristic prompts and lack of automatic prompt adaptation.":
            "We added matched instruction variants and an automatic finite instruction search. Seven predefined instructions are evaluated on 72 training-validation instances; the selector maximizes valid candidate count, then minimizes conditional mean candidate reference gap, then uses registry order for ties. Its choice is frozen before the final prompt-comparison test evaluation. "+prompt_change+" This establishes finite validation-only selection; open-ended instruction generation, DSPy, and adaptation to other scheduling variants were not tested.",
        "Dataset size and validation split.":
            "The audited release contains exactly 120,000 unique training-source instances and 1,800 unique held-out test instances, with no canonical overlap. The fresh run uses an explicit 118,000/2,000 training-validation partition; its 72 prompt-selection instances belong to that validation partition. The exact historical split identities were not recovered and are not asserted to match the fresh split.",
        "Discarded invalid solutions during testing.":
            f"The new first-instruction evaluation generates exactly {baseline['n_candidates']:,} candidates; {baseline['feasible_candidates']:,} pass strict validation. "+outcome+" Raw candidates and candidate/instance error counts are retained. The earlier counts of 27,960 generated and 1,750 valid candidates describe the historical 577-instance experiment and have not been revalidated here.",
        "Unclear source of performance improvements and missing ablations.":
            "The fresh controlled study isolates training progress along one Phi-3 run and introductory instruction changes at its final checkpoint, using matched instances and candidate budgets. Reference gaps are conditional on feasibility and are not optimality gaps. The earlier 1.2B/3.8B model-family and 8B representation comparisons are historical diagnostics, not controlled evidence about parameter count alone; their confounding factors remain stated. "+historical,
        "Comparison with lightweight neural models.":
            "The controlled checkpoint-and-prompt experiment does not itself compare against L2D or SLJ. Earlier average gaps use different feasible populations and candidate budgets, and the historical raw comparison evidence was not recovered. Any separate new baseline comparison requires its own verified matched-population results; checkpoint and prompt results alone do not establish competitive performance against specialized schedulers.",
        "Poor worst-case performance.":
            "The earlier Phi-3/SLJ maximum-gap values are retained as historical reported statistics; the underlying comparison records were not recovered and are not revalidated here. The new controlled experiment reports feasibility and conditional reference gaps without inferring comparative tail reliability against baselines that were not rerun.",
    }
    supplement_response=""
    if supplementary is not None:
        replacements["Comparison with lightweight neural models."]=(
            "A separate verified supplement compares four released L2D and SLJ checkpoints with the final Phi-3 checkpoint on the same 216 instances, with ten candidates per method and instruction. "
            "It reports full-population feasibility and reference gaps conditional on feasibility, with paired mean-gap comparisons on identical Phi-3-feasible subsets. "
            "These matched candidate counts do not imply equal compute, and this supplement does not revalidate the historical 577-instance comparison.")
        replacements["Poor worst-case performance."]=(
            "The earlier Phi-3/SLJ maximum-gap values are retained as historical reported statistics; the underlying comparison records were not recovered and are not revalidated here. "
            "The separate verified supplement reports observed maximum best-valid-of-ten reference gaps on each method's feasible subset. "
            "These sample maxima are conditional on feasibility and are not worst-case guarantees; comparisons on different feasible subsets do not establish comparative tail reliability.")
        supplement_response=(
            " A separate verified baseline supplement evaluates four released L2D and SLJ checkpoints on the same 216 frozen test instances, using ten candidates per method and instruction. "
            "Feasibility includes all 216 instances; the mean, median, and observed maximum reference gaps use each method's best valid candidate among ten, conditional on that method's feasible subset. "
            "Paired mean reference gaps are also reported on identical Phi-3-feasible subsets. "
            "Equal candidate counts do not imply equal compute, and an observed conditional sample maximum is not a worst-case guarantee.")
    for label,response in replacements.items():
        marker="\\item \\textbf{"+label+"}"
        if marker in rebuttal:
            start=rebuttal.index(marker);end=rebuttal.find("\n\n",start)
            if end<0:raise ValueError("Unexpected rebuttal paragraph structure")
            rebuttal=rebuttal[:start]+marker+" "+response+rebuttal[end:]
    additions=("The revision includes five controlled Phi-3 checkpoints, seven matched instruction strings, and validation-only instruction selection. "
               +outcome+" "+prompt_change+" One training seed and one paired decoding-seed scheme are used. "
               "The retained historical model-family, representation, 8B proxy, and discard statistics use their own populations; their denominators and trajectories are not combined with the new run."
               +supplement_response)
    for title in ("Additional Experiments","Controlled Phi-3 Revision Experiments"):
        marker="\\section*{"+title+"}"
        if marker in rebuttal:
            start=rebuttal.index(marker)+len(marker)
            match=re.search(r"\\(?:section\*\{|end\{document\})",rebuttal[start:])
            end=start+match.start() if match else len(rebuttal)
            rebuttal=rebuttal[:start]+"\n"+additions+"\n\n"+rebuttal[end:]
    if "\\section*{Controlled Phi-3 Revision Experiments}" not in rebuttal:
        rebuttal=rebuttal.replace("\\end{document}","\\section*{Controlled Phi-3 Revision Experiments}\n"+additions+"\n\n\\end{document}")
    return rebuttal


def integrate(study_dir,runs_dir,paper_dir):
    study_dir,runs_dir,paper_dir=map(Path,(study_dir,runs_dir,paper_dir))
    study,prepared=verify_study(study_dir)
    training=json.loads((runs_dir / "main/training_complete.json").read_text())
    run=json.loads((runs_dir / "main/run_manifest.json").read_text())
    if training["global_step"]!=1844 or run["status"]!="complete" or run["config"].get("pilot"):
        raise ValueError("The full, non-pilot Phi-3 training run has not completed")
    if run["train_data"]["records"]!=118000 or run["validation_data"]["records"]!=2000:
        raise ValueError("Training population differs from the planned 118k/2k split")
    expected_config={"total_steps":1844,"milestones":[0,461,922,1383,1844],"precision":"nf4-fp16",
                     "max_sequence_length":40000,"lora_rank":128,"lora_alpha":256,"lora_dropout":0.05,
                     "micro_batch_size":2,"gradient_accumulation_steps":32,"learning_rate":0.0002,
                     "warmup_steps":1,"seed":42,"eval_steps":25,"attention_implementation":"sdpa","group_by_length":True}
    if any(run["config"].get(key)!=value for key,value in expected_config.items()):
        raise ValueError("Training recipe differs from the frozen study")
    if set(run["lora_targets"]["selected"])!={"qkv_proj","o_proj"}:
        raise ValueError("Actual trainable LoRA modules differ from the recovered Phi-3 architecture")
    for split,key in [("train","train_data"),("validation","validation_data")]:
        if run[key]["sha256"]!=prepared["tokenization"][split]["sha256"] or file_hash(run[key]["path"])!=run[key]["sha256"]:
            raise ValueError("Training tokenization changed or differs from the frozen split")
    if not (study_dir / "selected_prompt.json").exists():
        raise ValueError("A prompt must be frozen before final test evaluation")
    selection=json.loads((study_dir / "selected_prompt.json").read_text())
    if selection["study_protocol_sha256"]!=study["protocol_sha256"] or selection["prompt_registry_sha256"]!=study["prompt_registry_sha256"]:
        raise ValueError("Prompt selection belongs to another study")
    prompts=[p.id for p in PROMPTS]
    val_manifest,_,val_records,_=completed_evaluation(runs_dir / "evaluations/final_prompt_validation",study_dir / "prompt_validation.jsonl",3,prompts)
    recalculated=select_validation_prompt(val_records,forbidden_test_instance_ids=[i.instance_id for i in load_dataset(study_dir / "test.jsonl")[0]])
    if selection["selected_prompt_id"]!=recalculated["selected_prompt_id"] or selection["validation_configuration_sha256"]!=val_manifest["configuration_sha256"]:
        raise ValueError("Frozen prompt selection does not match validation evidence")
    if Path(val_manifest["config"]["adapter"]).resolve()!=(runs_dir / "main/checkpoint-1844").resolve() or selection["validation_candidates_sha256"]!=file_hash(runs_dir / "evaluations/final_prompt_validation/candidates.jsonl"):
        raise ValueError("Prompt selection adapter or validation evidence changed")
    manifest,final,final_records,raw=completed_evaluation(runs_dir / "evaluations/final_prompt_test",study_dir / "test.jsonl",10,prompts)
    expected_adapter=(runs_dir / "main/checkpoint-1844").resolve()
    if Path(manifest["config"]["adapter"]).resolve()!=expected_adapter:
        raise ValueError("Final prompt evaluation did not use the fully trained checkpoint")
    if Path(run["config"]["base_path"]).resolve()!=Path(manifest["config"]["base_path"]).resolve() or val_manifest["base_artifacts"]!=manifest["base_artifacts"]:
        raise ValueError("Training and evaluations use different base models")
    evaluated_files={item["file"]:item["sha256"] for item in manifest["base_artifacts"]["files"]}
    trained_weights=run["base_model_files"].get("weight_file_sha256",{})
    if not trained_weights or any(evaluated_files.get(name)!=digest for name,digest in trained_weights.items()):
        raise ValueError("Training and evaluation base-weight content hashes differ")
    if (study_dir / "selected_prompt.json").stat().st_mtime > (runs_dir / "evaluations/final_prompt_test/evaluation_manifest.json").stat().st_mtime:
        raise ValueError("Prompt selection was modified after final test generation")
    checkpoint_rows=[]
    summaries={}
    for step in [0,461,922,1383]:
        cm,cs,cr,_=completed_evaluation(runs_dir / f"evaluations/checkpoint_{step}",study_dir / "test.jsonl",10,["seen_1"])
        if Path(cm["config"]["adapter"]).resolve()!=(runs_dir / f"main/checkpoint-{step}").resolve():
            raise ValueError("Checkpoint label/adapter mismatch")
        if cm["base_artifacts"]!=manifest["base_artifacts"]:
            raise ValueError("Checkpoint evaluations use different base model artifacts")
        summaries[str(step)]=cs["by_prompt"]["seen_1"]
        entry=json.loads((runs_dir / f"main/checkpoint-{step}/experiment_checkpoint.json").read_text())
        if entry["global_step"]!=step:
            raise ValueError("Training checkpoint receipt has the wrong step")
        hours=entry["cumulative_training_seconds"]/3600
        checkpoint_rows.append(table_row(f"{step} ({100*step/1844:.0f}\\%, {hours:.2f} h)",summaries[str(step)]))
    entry=json.loads((expected_adapter / "experiment_checkpoint.json").read_text())
    if entry["global_step"]!=1844:
        raise ValueError("Final checkpoint receipt has the wrong step")
    summaries["1844"]=final["by_prompt"]["seen_1"]
    checkpoint_rows.append(table_row(f"1844 (100\\%, {entry['cumulative_training_seconds']/3600:.2f} h)",summaries["1844"]))
    prompt_rows=[table_row(p.id.replace('_',r'\_')+(" (training wording)" if p.seen_in_training else " (unseen wording)"),final["by_prompt"][p.id]) for p in PROMPTS]
    truncations={p.id:sum(bool(row.get("hit_max_new_tokens")) for row in raw if row["prompt_id"]==p.id) for p in PROMPTS}
    selected=selection["selected_prompt_id"]
    selected_tex=selected.replace('_',r'\_')
    unseen_wording_list=(r"\begin{itemize}"+"\n"+"".join(
        "\\item \\texttt{"+p.id.replace('_',r'\_')+"}: \\emph{"+p.text+"}\n"
        for p in PROMPTS if not p.seen_in_training)+r"\end{itemize}"+"\n")
    delta=paired_bootstrap(final_records["seen_1"],final_records[selected])
    n=final["n_evaluated_instances"]
    hours=entry["cumulative_training_seconds"]/3600
    discard_sentences=[]
    for label,prompt_id in (("first training instruction","seen_1"),("validation-selected instruction",selected)):
        candidates=final["by_prompt"][prompt_id]
        total=candidates["n_candidates"]
        valid=candidates["feasible_candidates"]
        rejected=total-valid
        prompt_tex=prompt_id.replace('_',r'\_')
        discard_sentences.append(
            f"Under the {label} (\\texttt{{{prompt_tex}}}), {valid:,} of {total:,} generated candidates pass all strict validation checks; "
            f"{rejected:,} are rejected ({100*rejected/total:.2f}\\% of all {total:,} candidates).")
    discard_text=" ".join(discard_sentences)+" These candidate-level rejection rates include every generated candidate; instance feasibility requires only one valid schedule among ten."
    if selected=="seen_1":
        discard_text+=" The selector chose the first training instruction, so both descriptions refer to the same candidate pool."
    text=(r"\subsubsection{Controlled Phi-3 Training Budget and Prompt Experiments}"+"\n"
          r"\label{sec:phi3_revision}"+"\n"
          f"We conducted a fresh, controlled Phi-3-Mini-128K-Instruct run on 118,000 released training instances, with 2,000 held out for validation. "
          "The NF4 LoRA configuration uses rank 128, alpha 256, dropout 0.05, learning rate $2\\times10^{-4}$, effective batch size 64, and seed 42. "
          "We save the initialized adapter and checkpoints at 25, 50, 75, and 100\\% of the fixed 1,844-update schedule. "
          "The actual trainable modules are \\texttt{qkv\\_proj} and \\texttt{o\\_proj}; the listed \\texttt{fc1/fc2} names do not occur in this Phi-3 architecture. "
          "All checkpoint evaluations use the same instruction, instance order, paired decoding seeds, and exactly ten candidates per instance, with temperature 1.0, top-$k$ 50, and top-$p$ 0.95.\n\n"
          "For an instance with $n$ operations, each candidate has a maximum of $\\min(8192,48n+128)$ new tokens. Candidates reaching this limit remain in the feasibility denominator and are validated without repair. "
          f"The evaluation population contains {n} unique instances selected before generation, six from each of the 36 size groups in the released 1,800-instance test set. "
          "A complete audit found 120,000 unique training and 1,800 unique test matrices, no training--test overlap, and feasible solver labels for every record under the strict validator. "
          "The historical 577-instance manifest was not recovered; these are new paired results and are not replacements for or a reproduction of the historical comparison table. "
          "Recovered historical checkpoint metadata is inconsistent with an exact replay of the fresh split; its original split identities were not recovered, and we do not combine its trajectory with this run. "
          "For each feasible instance the reference gap is $100(C_{\\mathrm{best}}-C_{\\mathrm{ref}})/C_{\\mathrm{ref}}$, with the minimum valid generated makespan $C_{\\mathrm{best}}$ and released reference makespan $C_{\\mathrm{ref}}$. "
          "Reference optimality is not certified here. Conditional gap means may cover different feasible subsets and cannot alone establish a change in overall scheduling quality.\n\n"
          r"\begin{table}[!ht]\centering\small"+"\n"
          r"\caption{Controlled Phi-3 training-budget study. Feasibility includes all frozen test instances; brackets give 95\% Wilson intervals. Reference gaps are conditional on feasibility, using the best valid candidate among ten; reference optimality is not certified. Hours are active-run wall time including validation/checkpoint overhead, excluding setup and downtime between resumed processes.}"+"\n"
          r"\label{tab:phi3_training_budget}\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule"+"\n"
          r"Updates (budget, hours) & Feasible & Feasibility [95\% CI] & Mean reference gap & Median reference gap \\ \midrule"+"\n"+"".join(checkpoint_rows)+
          r"\bottomrule\end{tabular}}\end{table}"+"\n\n"
          "We hold the final model fixed and vary only the introductory instruction across the three training wordings and four predefined, unseen wordings. "
          "Automatic adaptation is a finite validation-only instruction search: each of seven candidate instructions receives three generated schedules on each of 72 instances drawn only from the 2,000 training-validation records, for 1,512 validation candidates. "
          "This population is disjoint from training and the final test source. The registry, selection rule, and populations were frozen before model outcomes. "
          "The selector maximizes valid candidate count, then minimizes conditional mean candidate gap, then uses the fixed registry order to break ties. "
          f"It selected \\texttt{{{selected_tex}}} before the final prompt-comparison test evaluation. "
          "Earlier checkpoint evaluations use the same frozen tests but do not enter the predetermined validation-only selector. "
          "The instruction list, selection evidence, raw candidates, strict validation messages, seeds, token counts, and times are retained with the experiment artifacts.\n\n"
          r"\begin{table}[!ht]\centering\small"+"\n"
          r"\caption{Prompt wording study with the fully trained Phi-3 checkpoint and the same ten-candidate test budget for every instruction. Reference gaps are conditional on feasibility, without established reference optimality; the validation-selected instruction uses one wording, without pooling test candidates across prompts.}"+"\n"
          r"\label{tab:phi3_prompt_wording}\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule"+"\n"
          r"Instruction & Feasible & Feasibility [95\% CI] & Mean reference gap & Median reference gap \\ \midrule"+"\n"+"".join(prompt_rows)+
          r"\bottomrule\end{tabular}}\end{table}"+"\n\n"
          "The four unseen introductory instructions are reproduced exactly below; the three training instructions are listed in the Training section.\n"+unseen_wording_list+"\n"
          f"Relative to the first training instruction, the validation-selected instruction changes test feasibility by {delta['difference_percentage_points']:.2f} percentage points "
          f"(paired instance-bootstrap 95\\% interval [{delta['ci95_percentile'][0]:.2f}, {delta['ci95_percentile'][1]:.2f}], 10,000 resamples). "
          "This finite search tests automatic selection within a fixed instruction family; it does not establish adaptation to arbitrary scheduling variants.\n\n"
          +discard_text+"\n\n"
          +"Generation-length limits can affect feasibility; per-instruction counts of candidates reaching the token cap are included in the retained results. "
          f"For the first training instruction, {truncations['seen_1']} of {n*10} candidates reached this cap; for the validation-selected instruction, {truncations[selected]} of {n*10} did so. "
          "The study uses one training seed and one paired decoding-seed scheme. The intervals describe instance-level uncertainty conditional on that run, not variability across training or decoding seeds. "
          "Checkpoint comparisons measure early stopping along one shared full-run learning-rate schedule, rather than separately optimized short-run schedules.\n")
    supplementary=supplementary_baselines(study_dir,runs_dir,final_records,final["by_prompt"],selected)
    if supplementary is not None:
        text+="\n"+supplementary["tex"]
    # Only verified artifacts reach this point. Preserve pre-revision source once.
    source=paper_dir / "neurips_2024.tex"
    anchor="\\subsubsection{Analysis of Discarded Solutions and Failure Modes}"
    include="\\input{revision_results}\n\n"
    original_paper=source.read_text()
    if anchor not in original_paper and include not in original_paper:
        raise ValueError("Manuscript integration anchor is absent; refusing an unlinked results artifact")
    paper=revise_publication(original_paper,summaries["1844"],delta,selected,hours,run)
    if include not in paper:paper=paper.replace(anchor,include+anchor)
    rebuttal_path=paper_dir / "rebuttal.tex"
    original_rebuttal=rebuttal_path.read_text() if rebuttal_path.exists() else None
    rebuttal=revise_rebuttal(original_rebuttal,summaries["1844"],delta,supplementary) if original_rebuttal is not None else None
    # All validation and rendering complete before any manuscript artifact is written.
    backup=paper_dir / "build/neurips_2024.before_experimental_integration.tex"
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists(): backup.write_text(original_paper)
    (paper_dir / "revision_results.tex").write_text(text)
    source.write_text(paper)
    if rebuttal is not None:
        rebuttal_backup=paper_dir / "build/rebuttal.before_experimental_integration.tex"
        if not rebuttal_backup.exists(): rebuttal_backup.write_text(original_rebuttal)
        rebuttal_path.write_text(rebuttal)
    atomic_json(study_dir / "integrated_results.json",{"status":"verified_results_integrated","checkpoint_summaries":summaries,
                "selected_prompt":selection,"prompt_feasibility_delta":delta,"prompt_summary":final,
                "prompt_candidates_hitting_token_cap":truncations,
                "supplementary_baselines":supplementary["summary"] if supplementary is not None else None,
                "report_code_sha256":file_hash(Path(__file__)),
                "tex_sha256":file_hash(source),"results_tex_sha256":file_hash(paper_dir / "revision_results.tex")})
    return {"status":"verified_results_integrated","paper":str(source)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir",type=Path,required=True)
    p.add_argument("--runs-dir",type=Path,required=True)
    p.add_argument("--paper-dir",type=Path)
    p.add_argument("--select-prompt-only",action="store_true")
    args=p.parse_args()
    result=select_prompt(args.study_dir,args.runs_dir) if args.select_prompt_only else integrate(args.study_dir,args.runs_dir,args.paper_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
