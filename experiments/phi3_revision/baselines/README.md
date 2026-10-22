This supplementary experiment evaluates the four authentic released policies
L2D 20x20 (durations 1..99), SLJ B32, B128 and B256 on the exact 216 frozen
Phi study instances, with ten stochastic candidates per policy and instance.
It does not alter the main Phi training, prompt registry, selection or decoding.

The independent protocol is `artifacts/PROTOCOL.json`. It binds input identities,
source and checkpoint hashes, dependency fingerprints, seeds, evaluation mode,
candidate budget and schedule reconstruction before baseline scientific outcomes.
All 216 inputs passed original SLJ feature finiteness and classical-route checks.
The protocol documents that released B256 has a different decoder architecture,
that exact baseline training seeds/manifests are unavailable, and that this
evaluation explicitly calls L2D `eval()` whereas original `testL2D.py` did not.

Use the existing legacy environment from the project root:

```sh
export PYTHONDONTWRITEBYTECODE=1
baseline_python=/home/lamsade/habgaryan/phd_project/SelfLabelingJobShop/slj/bin/python
"$baseline_python" -m unittest experiments.phi3_revision.baselines.test_baselines -v
"$baseline_python" experiments/phi3_revision/baselines/run.py verify
"$baseline_python" experiments/phi3_revision/baselines/run.py run --pilot --output-dir experiments/phi3_revision/baselines/artifacts/pilot
```

The pilot is an excluded synthetic 2x2 instance and produces 40 candidates.
Its receipt explicitly says `EXCLUDED_TECHNICAL_PILOT`; it cannot be used as a
scientific completion receipt. After independent review of the frozen protocol,
pilot evidence and resumption checks, production was launched on seacove on
2026-10-16 at 22:18 UTC. Its command is:

```sh
"$baseline_python" experiments/phi3_revision/baselines/run.py run --output-dir experiments/phi3_revision/baselines/artifacts/production
```

`artifacts/production.launch.json`, `production.pid`, `production.log` and
`production.exit` record the durable CPU process. Scientific completion requires
`artifacts/production/COMPLETE.json` and revalidation of every raw candidate;
the existence of a launch record alone does not establish completion.

Production exited successfully and all 8,640 candidates passed an independent
legacy-environment schedule audit. Its evidence is `artifacts/report_audit.json`.
Every policy scheduled all 216 instances. The manuscript now compares these
baselines with the verified update-1,383 (75%) Phi-3 checkpoint on that same
216-instance cohort; the full-training comparison remains pending. This cohort
is a stratified sample from the released test set, not a Taillard or DMU
benchmark evaluation. The comparison is interim evidence, not evidence that
Phi-3 training has completed.

Both pilot and production invocations resume the same output directory. A
completed instance is strictly revalidated before reuse. An interrupted instance
regenerates all ten candidates with the same per-instance seed, compares existing
raw evidence, and appends only missing slots. Each candidate retains raw actions,
native schedule arrays, a complete schedule, library makespan, strict validation,
timing, checkpoint hash and protocol identity. Completion requires every fixed
slot. The receipt records feasibility and conditional gaps against released
reference labels; reference labels are not assumed optimal.

L2D receives one-based machine IDs required by its original left-shift environment.
Its actual completion-time and machine-order arrays are retained and replayed;
append-only reconstruction would change L2D schedules. SLJ decisions are captured
by a temporary in-memory observer of the original scheduler. Its uniquely forced
last action is restored to the original matrix after verification, and native
append schedules are replayed with integer durations. Neither original library
source nor its checkpoints are modified.

The CPU-only run uses one Torch thread. Ten candidates match the Phi candidate
count, not FLOPs or runtime. SLJ uses one batch of ten, L2D ten sequential
trajectories. Batch and individual times remain distinguishable in raw files.

`report.py` is a read-only reporting helper. It rechecks source, model, parent
study and raw-file hashes, then replays the scheduling evidence in the original
legacy Python environment. It returns full-population metrics and paired gaps
on identical Phi-feasible subsets. The main study reporter calls it only after
the full Phi-3 study and validation-only prompt selection have been verified.
