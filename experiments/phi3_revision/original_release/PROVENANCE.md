# Original release and revision experiment provenance

The paper's public source is [starjob42/datasetjsp](https://github.com/starjob42/datasetjsp/tree/24096f99b4961cc322dedfa74e3de2d811631080), frozen at commit `24096f99b4961cc322dedfa74e3de2d811631080`. Copies in this directory preserve the downloaded source files. `release_manifest.json` records their URLs, sizes and SHA256 hashes.

## Verified public data

`jssp_llm_format_120k.json` contains 120,000 unique problems across 50 sizes, with 2,400 problems per size. Its 1,136,426,260 bytes match the Git LFS SHA256 `dd09b81f82fcb1549466888b82aa1086b783ffface3e821ca63cb46072647796`. The operation durations actually range from 1 to 499; the manuscript describes 5 to 500. The original verbose machine descriptions include operation indices and therefore encode job precedence. Later compact machine descriptions found in shared storage omit those indices and cannot replace these inputs in a faithful experiment.

Despite its filename, `test_2000.json` contains 1,800 problems: 50 in each of 36 sizes from 2×2 through 9×9. Every problem has fewer than 100 operations, so a size filter cannot recover the manuscript's reported 577 instances. Its durations range from 1 to 98, rather than the manuscript's stated 1 to 199. SHA256: `2b473b9afa7661004195853c2f3cbae72de22470bea9a582a2f5d70487c5bcf3`.

The complete audit in `audit.json` confirms that all 120,000 training-source schedules and all 1,800 test schedules satisfy the strict revision validator. There are no duplicate canonical problems and no train/test overlap. Canonical identity uses the dimensions, routes and durations, excluding reference schedules and makespans. These checks establish feasibility, not optimality. The audit can be rerun with `python3 experiments/phi3_revision/original_release/audit_original_release.py`.

## Recovered model and protocol discrepancies

The released inference script names `microsoft/Phi-3-mini-128k-instruct` with `checkpoint-1750`. A matching-name adapter was recovered from `/mnt/volatile/users/habgaryan/litgpt/efficient_llm_jssp/checkpoint-1750/`. It uses rank 128, alpha 256, dropout 0.05 and the manuscript's four adaptation modules. Its adapter SHA256 is `e5581b6b763bf52c62b04602f19a5208314c353313472e9cee7c579de3429e2a`. Detailed hashes and sanitized settings appear in `recovered_checkpoint_provenance.json`.

That saved state records one epoch, 1,799 planned optimizer steps, and step 1,750 at epoch 0.9724. Those steps and the recorded evaluation throughput are consistent with approximately 115,000 training and 5,000 validation examples; the precise historical split was not recovered. The public training script specifies a 118,000/2,000 split but two epochs, while the manuscript specifies that split and one epoch. Saved logging/evaluation intervals are 50 steps; the manuscript says 25. The public checkpoint archive link returned HTTP 404 during retrieval, so binary equivalence between the recovered adapter and published archive remains unverified. Other surviving Phi adapters use Phi-3.5 with rank 64 and are separate experiments.

Neither the original 577-instance manifest nor its complete Phi-3 candidate records was located. The 998-record test file and checkpoint-1924 results in the later `starjob_complete` project describe another configuration and population; they cannot establish the original Phi-3 result provenance.

## Validation and the fresh controlled study

The original released validator checks machine allocation, duration, machine overlap and precedence for the operations it parses. It does not require every operation exactly once, check nonnegative start times and timing arithmetic, or verify the reported makespan against all operations. The revision uses the independent strict validator in `../core.py`; the full data audit above passed its stronger checks.

The revision freezes a fresh 118,000/2,000 split and trains the complete 118,000-example set for one epoch, retaining steps 0, 461, 922, 1383 and 1,844. This follows the manuscript recipe and does not claim an exact replay of the recovered historical run. `../study/STUDY.json` fixes a 216-instance test subset, six problems per released test size, and a separate 72-instance prompt-validation subset, two per size from the held-out 2,000 examples. Salted canonical hashes determine membership without inspecting reference labels or model outcomes. The study also freezes the seven-prompt registry, decoding budgets and the validation-only prompt-selection rule. New measurements must be described as this controlled study, rather than evaluations on the unrecovered historical 577-instance subset.

Separate technical pilots test model loading, generation and checkpoint recovery. Their candidates are excluded from reported experimental counts. A shortest-prompt smoke test preceded the freeze; neither membership nor instructions were changed using its outcome. Any pilot instance included by the fixed hash rule is evaluated again under the frozen study conditions.
