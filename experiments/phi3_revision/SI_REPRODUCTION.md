# Online Resource 1: setup, provenance and independent reproduction

Read the archive-root README.txt first for the article, author/contact metadata
and whether this particular archive contains final scientific outputs. The
archive preserves the original `experiments/phi3_revision/` module hierarchy.
`MANIFEST.json` hashes every other archive member and separates original file
copies from generated packaging documentation. The frozen scientific Python,
JSON and JSONL files are copied byte for byte.

## CPU-only inspection without dependencies, models or server access

Extract the entire ZIP into an empty directory. Change to its root (the
directory containing README.txt and experiments/), then run:

```sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/experiments/phi3_revision"
python3 -B experiments/phi3_revision/si_readiness.py --check-help
python3 -B -m experiments.phi3_revision.prepare --help
python3 -B -m experiments.phi3_revision.freeze_study --help
python3 -B -m experiments.phi3_revision.train --help
python3 -B -m experiments.phi3_revision.evaluate --help
python3 -B -m experiments.phi3_revision.run_study --help
python3 -B -m experiments.phi3_revision.auto_prompt.protocol --help
python3 -B -m experiments.phi3_revision.auto_prompt.optimizer --help
python3 -B -m experiments.phi3_revision.auto_prompt.evaluate --help
python3 -B -m experiments.phi3_revision.auto_prompt.report --help
python3 -B -m experiments.phi3_revision.auto_prompt.publish --help
python3 -B -m experiments.phi3_revision.baselines.run --help
```

These commands use Python 3.11 or newer and its standard library only. The
readiness helper verifies archived bytes, selected frozen protocol bindings and
the command parsers; it never loads Torch, a tokenizer or a model, performs
inference/training, contacts a server, or changes a file. Successful `--help`
does not certify scientific reproducibility, sufficient GPU memory or completed
experiments. The `original_release/audit_original_release.py` program is **not**
a help entry point: executing it audits large inputs and writes derived files.

## Recorded environments and externally required inputs

Archive-root ENVIRONMENT.json records versions from the actual main-run manifest
and the separately frozen legacy-baseline protocol. These are execution records,
not an installation lock file or an assertion that all transitive packages and
drivers were fingerprinted. Main training used Torch 2.5.1, Transformers 4.57.1,
PEFT 0.18.1, Accelerate 1.4.0 and bitsandbytes 0.45.5. Bitsandbytes was available
in the original project's vendor/ directory; binary vendor files are excluded
from this archive. Provide these packages in an independently prepared CUDA
environment before any model operation. No installation is performed by the
inspection commands. Baseline evaluation used a **different** environment:
Torch 1.13.1, NumPy 1.26.4, torch-geometric 2.2.0, torch-scatter 2.1.2,
torch-sparse 0.6.18 and Gym 0.26.2; its protocol retains dependency fingerprints.
Do not substitute that environment for the Phi-3 environment.

PROVENANCE.json contains pinned source/data/model identities and content hashes:

- Dataset and historical code: https://github.com/starjob42/datasetjsp at
  commit `24096f99b4961cc322dedfa74e3de2d811631080`.
  `original_release/release_manifest.json` supplies exact URLs, byte sizes and
  SHA256 values. The 120,000-record training file uses Git LFS and is about
  1.14 GB; the released `test_2000.json` actually contains 1,800 records.
  The original README records CC BY-SA 4.0 for the dataset; observe its terms.
- Official base: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct at
  revision `f3c06aed622e14ca0abf5115094e4fc9a9948f36`. Obtain the complete model
  and tokenizer snapshot and check its file hashes against PROVENANCE.json.
  Weights total about 7.64 GB and are excluded. Observe the upstream license.
- Baseline source: https://github.com/AndreaCorsini1/SelfLabelingJobShop at
  commit `f5b54d789cbb0a11c6bb5bedeccefddb04e2b8c6`. Its released L2D and SLJ
  checkpoint identities and source hashes are retained in the baseline protocol.
  Obtain those external sources/weights and preserve their licenses separately.
- The locally recovered historical checkpoint-1750 is not interchangeable with
  the fresh controlled checkpoints. Its public archive was unavailable during
  recovery, so binary equivalence to that public archive remains unverified.

This small archive excludes full training data, tokenized training examples,
model/adapter weights, optimizer and RNG binaries, external baseline libraries,
binary dependencies, and the local LaTeX compiler/cache. It includes the frozen
216-instance test and a 72-instance prompt-selection subset of the 2,000-record
training-validation partition. The prompt-selection subset contains exactly
two instances from each of 36 size groups; its ordered records are in
`experiments/phi3_revision/study/prompt_validation.jsonl` with SHA-256
`3bca79615e7b7b3310fad171fa9a4d5f5c057bff535bb4fc2532979f9d0e495e`.
Inclusion of those splits does not reconstruct the unrecovered historical
577-instance evaluation.

## Larger-instance held-out diagnostic

The separate `study/large_generalization_26_per_shape.jsonl` diagnostic contains
26 records for each of 14 larger shapes (10x2 through 10x10, 12x12, 15x10,
15x15, 20x15, and 20x20), for 364 instances and 3,640 fixed candidate slots.
All records belong to the recorded 2,000-example training-validation partition
and are disjoint from the 118,000 training records. Twenty-six per shape is the
largest balanced sample possible because the held-out 10x4 shape contains only
26 records. This is a separate held-out generalization diagnostic, not a
replacement for or pooled analysis with the manuscript's 216-instance test.
The exact selection, split membership, and hashes are recorded in
`study/LARGE_GENERALIZATION_HELDOUT_EVAL.json`.

To run the prepared 100%-checkpoint, fixed-`seen_1` diagnostic after reserving
an appropriate GPU:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m experiments.phi3_revision.evaluate \
  --base-path experiments/phi3_revision/models/phi3-mini-128k \
  --adapter experiments/phi3_revision/runs/main/checkpoint-1844 \
  --data experiments/phi3_revision/study/large_generalization_26_per_shape.jsonl \
  --prompt-ids seen_1 --candidate-count 10 --batch-size 5 --seed 42 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --output-dir experiments/phi3_revision/runs/evaluations/large_generalization_26_per_shape
```

The external benchmark diagnostic is separate from that validation-partition
diagnostic. `study/taillard_dmu_50x20.jsonl` contains all 150 available
Taillard and DMU instances with no dimension exceeding 50x20: 70 Taillard
records and 80 DMU records, ten per listed benchmark group. The Taillard 100x20
group is excluded. The source files, hashes, imported benchmark schedules, and
the project-validator checks are recorded in
`study/TAILLARD_DMU_50X20_EVAL.json`. Known benchmark makespans remain in the
matrix reference trailers; imported schedules are retained only as provenance
and were independently checked for feasibility. This diagnostic is reported
separately from the 216-instance study and is not a claim of performance on
Taillard or DMU until the model evaluation is completed.

## Immutable original evidence versus a new reproduction elsewhere

The original protocols intentionally contain absolute paths, source hashes,
input hashes, host/resource requirements and original execution receipts. They
remain unedited here. The read-only readiness helper checks portable byte
bindings without pretending those original absolute paths exist locally. The
scientific verifiers require the original dependencies/inputs and original
path identities; simply changing a path inside a frozen JSON invalidates its
hash. Moving this archive to another machine cannot retain the same complete
execution protocol identity.

A new execution needs its **own new output directories, new preparation and
protocol freeze, independently checked resources and separately identified
receipts**. Preserve this archived evidence separately. Resolve all inputs
locally, compare pinned source/data/model hashes, then use the included CLI
options. For example, after providing the external corpus and tokenizer, these
commands illustrate the preparation order; they are not performed by this
archive, and the placeholder paths must be replaced:

```sh
python3 -B -m experiments.phi3_revision.prepare \
  --training-source /YOUR/INPUT/jssp_llm_format_120k.json \
  --test-source /YOUR/INPUT/test_2000.json \
  --tokenizer /YOUR/MODEL/phi3-mini-128k \
  --output-dir /YOUR/NEW/REPRODUCTION/data
python3 -B -m experiments.phi3_revision.freeze_study \
  --prepared-manifest /YOUR/NEW/REPRODUCTION/data/manifest.json \
  --prepared-test /YOUR/NEW/REPRODUCTION/data/test.jsonl \
  --output-dir /YOUR/NEW/REPRODUCTION/study
```

The prepared split is 118,000 training and 2,000 validation examples. Verify
its canonical identities and outcome-blind 216/72 partition against archived
membership before comparison. On an independently verified available GPU with
at least 40 GiB memory, `train.py` accepts the new base/data/output paths. Its
defaults specify one 1,844-update epoch and checkpoints at 0/461/922/1383/1844;
pilot outputs cannot establish production completion. NF4 training and the
fresh inference precision are different: FP16 base inference retains PEFT's
FP32 LoRA tensors and corresponding adapter-path arithmetic.

`run_study.py` is the original multi-server deployment manager: its default
workers, interpreter, SSH hosts and GPU binding match the original deployment.
`auto_prompt.optimizer` and `auto_prompt.publish` also contain original server,
compiler and path restrictions. Their `--help` is portable; their execution is
not a universal local launcher. Review these constraints before preparing a new
deployment, and identify any deployment code change in a newly frozen protocol.
Do not execute the archived defaults against the authors' servers.

For manual new-run evaluation, `evaluate.py` accepts local `--base-path`,
`--adapter`, `--data`, `--prompt-ids` and `--output-dir`. Keep the frozen seed,
temperature1, top-k50, top-p0.95, candidate counts and length-cap rule. Evaluate
all checkpoint arms on the identical 216 inputs. Evaluate all seven final-model
instructions on the separate 72 inputs at three candidates each, freeze the
validation-only choice, and only then generate final prompt-test arms at ten
candidates per input. Do not choose prompts using test results. The separate
automatic-refinement study requires its own newly frozen protocol, engineering
evidence, validation feedback chronology and selected instruction before fresh
test generation; its protocol CLI and README describe those barriers.

These directions document inputs and ordering. They do **not** assert that the
training or evaluations have been replayed on another machine. Different
hardware, CUDA libraries or dependency versions can change stochastic outputs.
The included historical READMEs describe the original deployment; this document
supersedes their local-path setup advice for extracted-archive inspection.

## Scientific records and final archive paths

All scientific paths below start with `experiments/phi3_revision/`:

- `study/`: frozen 216/72 membership, prompt registry and preparation split.
- `data/manifest.json`: original preparation, source hashes and index split;
  full tokenized data are excluded.
- `runs/evaluations/{checkpoint_0,checkpoint_461,checkpoint_922,checkpoint_1383,
  final_prompt_validation,final_prompt_test}/`: in a **final** archive, raw
  candidates, configuration manifests, summaries and extensionless COMPLETE
  receipts. Runtime records remain distinguishable from scientific outcomes.
- `runs/main/`: in a final archive, training completion, manifest, arguments,
  milestone metadata and tokenizer/configuration files, without weight/state
  binaries. `runs/manager/` retains the completion/plan/status evidence.
- `study/selected_prompt.json` and `study/integrated_results.json`: in a final
  archive, the primary validation-only choice and verified measured report.
- `baselines/artifacts/production/`: in a final archive, four policies' raw
  schedules and completion evidence on the same 216-by-ten population.
- `auto_prompt/artifacts/production/`: in a final archive, generated proposal
  attempts, all five 72-by-three validation evaluations, SELECTED.json frozen
  before the selected instruction's 216-by-ten test, COMPLETE.json and
  INTEGRATED.json. The separate `artifacts/publishing/` receipts bind final
  frozen-carrier publication; journal bridge copies are editorial derivatives.

A pending archive contains the methods, split files and original protocols,
**not** these unfinished primary/automatic result populations. A completed
baseline run alone does not establish main-training or automatic-study completion.
Final scientific verification must revalidate raw candidates and chronology using
the corresponding original input/model/dependency evidence; archive byte checks
alone are insufficient. Paper typesetting sources and the compiled article are
provided separately in the flat journal-source ZIP.
