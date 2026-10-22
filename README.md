# LLMs Can Schedule

Clean, source-only code for the Phi-3 job-shop-scheduling experiments.

- **Dataset repository:** https://github.com/starjob42/datasetjsp
- **Code branch:** `code_reproduction`
- **Checkpoints:** published separately on Hugging Face; add the final model
  link here once the repository is created.

This repository deliberately excludes trained weights, large datasets,
evaluation outputs, caches, and publication artifacts. The frozen
experiment definitions and source-level tests are included.

## Quick start

Clone the branch and enter the project:

```bash
git clone --branch code_reproduction \
  https://github.com/starjob42/llms_can_schedule.git
cd llms_can_schedule
```

Create an environment with Python 3.11 or newer and install the dependencies
listed in `experiments/phi3_revision/original_release/requirements.txt`.
For GPU training/inference, install a CUDA-compatible PyTorch build and the
matching PEFT/Transformers versions required by the experiment environment.

Run the source-level checks without downloading model weights or datasets:

```bash
python3 -m compileall -q experiments
python3 -m unittest discover \
  -s experiments/phi3_revision -p 'test_*.py'
```

## Reproduce the experiment

The complete dataset and dataset tooling are maintained in
[starjob42/datasetjsp](https://github.com/starjob42/datasetjsp). Obtain the
dataset and the official Phi-3 base model separately; do not commit either to
this repository.

The reproducibility workflow is:

1. Prepare and audit the data with `experiments/phi3_revision/prepare.py`.
2. Freeze the study populations with `freeze_study.py`.
3. Train the LoRA adapter with `train.py`.
4. Run fixed-budget generation with `evaluate.py`.
5. Validate and aggregate results with `report.py`.

The frozen protocol is recorded in
`experiments/phi3_revision/study/STUDY.json`; prompt wording and split metadata
are in the other files under `experiments/phi3_revision/study/`.
`experiments/phi3_revision/README.md` documents the full protocol, expected
completion markers, GPU requirements, and independent raw-output audit.

Do not treat a PDF, partial run, pilot, or missing `COMPLETE` marker as a
completed scientific result.

## Repository layout

```text
experiments/phi3_revision/
  core.py                 # parsing, identities, and strict validation
  prepare.py              # data preparation and split audits
  train.py                # reproducible LoRA training
  evaluate.py             # resumable fixed-candidate evaluation
  report.py               # guarded result aggregation
  study/                  # frozen protocol definitions
  test_*.py               # source-level and protocol tests
```

## Model checkpoints

Checkpoint publication is intentionally separate from this source repository.
When published, use the exact Hugging Face model identifier and revision in
the links below, and verify the adapter checksum before evaluation:

- 25% (`checkpoint-461`): **to be added**
- 50% (`checkpoint-922`): **to be added**
- 75% (`checkpoint-1383`): **to be added**
- 100% (`checkpoint-1844`): **to be added**
