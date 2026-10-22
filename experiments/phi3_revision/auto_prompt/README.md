# Separate automatic instruction generation and refinement

This separate experiment performs automatic validation-guided refinement of one
global scheduling instruction. It leaves the primary seven-prompt study, its
frozen inputs and all baseline sources unchanged.

The initial instruction is the original `seen_1`. An unadapted, pinned Phi-3 base
generates two new instructions in each of two rounds. Both proposals in a round
receive the same incumbent and aggregate validation feedback. All five
instructions receive fresh evaluations on the same 72 held-out instances with
three candidates each: exactly 1,080 validation schedules. Selection maximizes
feasible candidates, then minimizes conditional mean reference gap, then uses
creation order. The selected instruction is frozen before a fresh evaluation on
the existing 216 test instances with ten candidates each: exactly 2,160
additional test schedules. Only after this new test completes are the parent
test outcomes opened for matched descriptive comparisons with the original
instruction and the original fixed-registry selector. The separate study does
not evaluate DSPy, per-instance adaptation, or transfer to other variants.

`optimizer.py` owns proposal generation, format checking, round selection and
durable orchestration. `evaluate.py` uses an isolated copy of the unchanged
native evaluation module. `protocol.py` freezes the separate protocol;
`report.py` verifies raw artifacts before paper integration.

Proposals use native chat and the unadapted FP16 base. Acceptance checks only
format: one strict JSON object, exactly one `instruction` field, a clean single
line with 20--600 printable ASCII characters and at most 128 official-tokenizer
tokens, no role delimiters, and normalized uniqueness. Accepted text is preserved
exactly. Every attempt, raw output, message, seed and rejection is saved. Each
slot has at most four attempts; exhaustion or generation failure stops the run
without a manual fallback. The proposer sees the incumbent and validation
aggregates/error codes; it receives no problem instances or raw schedules.

Run all focused CPU checks from the project root; they instantiate no GPU model:

```sh
PYTHONPATH=experiments/phi3_revision python3 -m unittest \
  auto_prompt.test_protocol auto_prompt.test_optimizer \
  auto_prompt.test_evaluate auto_prompt.test_report
```

After independent engineering review and protocol freeze, the default invocation
only reports readiness:

```sh
/home/lamsade/habgaryan/miniconda3/envs/unsloth_env/bin/python -m experiments.phi3_revision.auto_prompt.optimizer --protocol experiments/phi3_revision/auto_prompt/artifacts/PROTOCOL.json
```

The production invocation adds `--execute`. It waits for the complete
primary manager and full 1,844-step training, then holds its own advisory lock
and checks the specified seacove A6000 is idle before every GPU child. A separate
proposal child exits after each round before any schedule evaluation child starts,
releasing the proposal model and CUDA context. Runtime `RUN.json` binds the actual
completed adapter and primary completion hashes.

Engineering pilots use excluded synthetic instances or synthetic aggregate
feedback. They do not provide scientific validation/test results. Early format
failures remain preserved as engineering evidence. This is a finite automatic
global-instruction refinement experiment; it is not per-instance adaptation.

`protocol.build_protocol(...)` writes nothing. Without engineering evidence it
returns `purpose: ENGINEERING_ONLY` and `status: engineering_draft_excluded`.
Only explicitly excluded `pilot=True` evaluation may load that draft without
full file verification. Scientific optimizer, evaluation, and reporting require
full verification and reject engineering drafts. Toy pilot instances must be
outside both scientific populations; draft metadata is not a protocol freeze.

After engineering checks and native pilots succeed, rebuild with evidence JSON
files containing `status: "verified"`, `purpose: "ENGINEERING_ONLY"`, and a
nonempty `checks` object whose values are all `true`. `freeze_protocol` binds
their exact file hashes, all participating source files, the parent protocol and
split, released source/tokenization files, and base-model content hashes. It
refuses retrospective freezing over scientific outputs and replacement of a
different frozen protocol. Final freezing follows stable code and pilot evidence.

Scientific artifacts under the frozen `output_dir` include:

- `instructions/{id}.json`: exact text/hash, origin, token count, creation order.
- `proposals/round_R_slot_S/attempt_A.json`: immutable attempts with timestamps,
  exact messages, seeds, raw outputs, format decisions, and self-hashes.
- `evaluations/validation/{id}/`: five independent 72-by-3 evaluations.
- `feedback/{id}.json` and `rounds/round_R.json`: verified scores and decisions.
- `SELECTED.json`: all validation receipts, scores, and frozen selected text.
- `evaluations/test/{selected_id}/`: only the selected instruction, 216-by-10.
- `COMPLETE.json`: final scientific receipt and evidence file hashes.

`report.build_comparison(protocol_path)` is read-only. It replays proposal
messages, format decisions, and round/final choices, revalidates schedule
outputs, verifies source/model/parent hashes and selection chronology, and
then computes paired instance-bootstrap feasibility differences. Gaps use
released feasible reference labels without certified optimality and remain
conditional on feasibility.

Publication integration is explicit:

```sh
python -m experiments.phi3_revision.auto_prompt.report \
  --protocol PROTOCOL.json --paper-dir PAPER_DIR
```

It requires fully verified additional results and the completed parent
integration. All rendering finishes before any publication write. Exact raw
instruction text stays in the artifacts; TeX escaping handles each original
special character once. Integration adds separate method/results tables and
updates the abstract, introduction, conclusion, limitations, and response using
measured outcomes, including declines or zero-feasibility results. Equal
per-test candidate budgets do not imply equal total prompt-search compute.

The reviewed protocol is now frozen in `artifacts/PROTOCOL.json` with digest
`2a6b4408c73a7c5f6a93be3500d7af1aa07279717ea05941db55e8d6b8bb4479`.
`artifacts/ENGINEERING_REVIEW.json` binds the 49 passing CPU checks, native
proposal and evaluation pilots, worst-context memory check, and synthetic
formatting evidence. These checks are excluded from scientific results.

The durable production command is the separately reviewed publishing driver:

```sh
/home/lamsade/habgaryan/miniconda3/envs/unsloth_env/bin/python -B -u -m experiments.phi3_revision.auto_prompt.publish --protocol experiments/phi3_revision/auto_prompt/artifacts/PROTOCOL.json --execute --poll-seconds 60
```

It has been launched on seacove; `artifacts/PUBLISH_LAUNCH.json` records the
command, PID, protocol, driver and native publication-smoke hashes. Monitor
`artifacts/publish.pid`, `artifacts/publish.log`,
`artifacts/production/STATUS.json`, and `artifacts/publishing/STATUS.json`.
The process waits for full primary completion, runs the refinement study,
revalidates and integrates measured results, then compiles and inspects both
documents before replacing the paper's root PDFs. Its publication receipt is
`artifacts/publishing/COMPLETE.json`; failures are preserved under the publishing
attempt directory and `artifacts/publishing/EXIT.json`. Direct process and GPU
checks are still required when monitoring: a status file alone is not proof of
a live job. A final independent scientific and publication audit remains
required after the driver completes.
