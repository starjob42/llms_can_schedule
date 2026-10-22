# Phi-3 revision experiments

This directory contains a fresh controlled reproduction of the manuscript's
Phi-3 recipe and the requested prompt experiments. The historical 577-instance
manifest has not been recovered. New results therefore use a frozen, balanced
216-instance subset of the authentic released 1,800-instance test file. They
must be identified as new experiments, rather than merged into the historical
577-instance comparison.

The complete 120,000-example release and every reference schedule passed the
strict data audit: 120,000 unique training instances, 1,800 unique test
instances, no duplicates, no training/test overlap, and no invalid labels.
The original release commit, content hashes, recovered checkpoint evidence,
and protocol discrepancies are recorded in `original_release/`.

## Frozen experiment

- Fresh NF4 Phi-3-Mini-128K LoRA training on all 118,000 examples, with a
  separate 2,000-example validation split. One epoch is 1,844 optimizer updates;
  the last update uses the remaining 48 examples. No small-data substitute is
  permitted.
- Initialized, 25%, 50%, 75%, and 100% checkpoints share one data order and
  learning-rate schedule. Rank 128, alpha 256, dropout .05, microbatch 2,
  accumulation 32, learning rate .0002, seed 42.
- Actual adapter targets are `qkv_proj` and `o_proj`. The `fc1` and `fc2` names
  listed in the manuscript are absent from this Phi-3 architecture; the recovered
  released adapter also contains only attention-module LoRA tensors.
- Identical 216 test instances, ten candidates each, temperature 1.0, top-k 50,
  top-p .95, FP16 inference, and paired decoding seeds for every comparison.
- The final checkpoint is evaluated on three training instructions and four
  predefined unseen instructions.
- Fixed-registry prompt selection compares those seven instructions on 72 separate
  validation instances, with three candidates per instruction and instance.
  The selected instruction is frozen before final test generation.

`study/STUDY.json` and its hashed inputs fix the populations and protocol.
`study/PROMPTS.json` contains the exact wording. Native assistant-generation
framing is explicitly recorded; it differs from the old released inference
utility. All new arms share the corrected framing.

## Programs and evidence

- `prepare.py`: immutable split, canonical overlap checks, and tokenization.
- `freeze_study.py`: outcome-blind stratified test/validation selection.
- `train.py`: GPU and data preflight, full training, milestone receipts,
  resumable optimizer/scheduler/RNG states, and completion markers.
- `evaluate.py`: fixed candidate budgets, raw outputs, strict validation,
  progress recovery, explicit incomplete populations, and completion markers.
- `run_study.py`: durable training/evaluation orchestration on the authorized
  servers; its `--execute` option is required to launch work.
- `report.py`: validates completed non-pilot evidence, freezes validation-only
  prompt selection, and writes the measured tables into the requested paper.
  It refuses missing, skipped, partial, pilot, or mismatched experiments.
- `baselines/`: a separately frozen comparison of released L2D and SLJ policies
  on the same 216 instances and ten-candidate budget. All 8,640 schedules have
  completed and passed an independent raw-output audit. Final integration also
  reports paired reference gaps on each Phi-3-feasible subset; the comparison
  waits for verified final Phi-3 outputs. See `baselines/README.md`.
- `auto_prompt/`: an additional required experiment that automatically proposes
  four instructions in two validation-guided rounds. It evaluates five
  instructions on 72 held-out instances with three candidates each, freezes its
  choice, and then generates ten fresh candidates on each of the 216 test
  instances. Its separate runner waits for the primary study and full training
  to finish before using the seacove A6000. See [auto_prompt/README.md](auto_prompt/README.md).

CPU checks:

```bash
python3 -m unittest discover -s experiments/phi3_revision -p 'test_*.py'
```

The production Python environment is
`/home/lamsade/habgaryan/miniconda3/envs/unsloth_env/bin/python`.
Bitsandbytes .45.5 is installed only in this project's `vendor/` directory;
add its absolute path to `PYTHONPATH`. The pinned official base model is in
`models/phi3-mini-128k/`, revision recorded in `artifacts/base_revision.txt`.

On seacove, bind the A6000 by UUID
`GPU-025e10e6-263f-d814-6dd5-added86fc8af`. Bare CUDA index 3 can select a 2080 Ti
unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. The trainer requires at least
40 GiB of device memory and exactly one visible GPU.

Engineering pilots are explicitly marked and excluded from manuscript tables.
Training completion requires `runs/main/training_complete.json`; evaluation
completion requires each job's `COMPLETE` marker and revalidated raw candidates.
The manager's status file and job logs record ongoing work and failures.
`study/integrated_results.json` is produced only after all required experiments
in the primary study have completed and their measured results have been
integrated. The overall revision also requires the separate automatic-refinement
run and its verified paper integration; the primary manager's `COMPLETE` marker
alone does not establish that the user's task is complete.

For one read-only live snapshot on the manager host, run
`python3 -B experiments/phi3_revision/monitor_live.py --once`. It checks actual
process identities, rejects zombies, distinguishes transient SSH uncertainty
from confirmed failure, and reports validation and candidate progress. In Codex
use authoritative host execution (`sandbox_permissions=require_escalated`);
the sandbox's process namespace cannot establish host process liveness. This
snapshot does not replace the final independent raw-result audit.

The real baseline manuscript has already been compiled and checked at
`../../LLMs_can_Schedule_paper/neurips_2024.pdf`. Its original invalid PDF is
preserved under that paper's `build/` directory. A compiled PDF alone does not
mean the new experiments have completed; consult the experiment markers above.

Final verification must check the completed training and every required raw
evaluation population, the frozen validation-only selection, and the measured
tables against `study/integrated_results.json`. The manager compiles the main
manuscript. Also compile the integrated `rebuttal.tex` with the same cached
Tectonic 0.16.9 environment and inspect both final PDFs, including their current
headline values and comparison tables. Synthetic formatting fixtures under
`/tmp` and `artifacts/publication_preflight.json` verify reporting machinery;
they cannot establish completion of the Phi-3 experiments.

After primary completion, also require `auto_prompt/artifacts/PROTOCOL.json`
and the automatic-refinement output's `COMPLETE.json`, `SELECTED.json`, and
`INTEGRATED.json`. Revalidate its 1,080 validation candidates and 2,160 fresh
test candidates, integrate `auto_prompt_results.tex`, then compile and inspect
the manuscript and rebuttal again. The final publication must contain the
measured checkpoint, prompt-wording, released-policy baseline, and automated
instruction-refinement results with their distinct evaluation populations and
limitations stated explicitly.
