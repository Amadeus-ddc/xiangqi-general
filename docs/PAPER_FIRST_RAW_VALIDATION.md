# First formal curriculum validation

The fresh full-width run completed its first 2,000 optimizer updates and its first raw current-static validation on 2026-10-15. This is an early course checkpoint, not completed curriculum training, playing-strength acceptance or coaching acceptance.

The [complete validation readback](../evidence/paper-course-v5-first-raw-validation-v1.json) independently regenerated all 700 grading records from the retained model responses and checked the declared questions, order and question variants. Every trainable adapter tensor is finite FP32 and bitwise equal to the corresponding tensor in the actual step-2000 training checkpoint: 671,806,496 parameters in total. The [resumption readback](../evidence/paper-course-v5-first-validation-resumption-v1.json) separately verifies the actual four-rank continuation from step 2000, with unchanged source, configuration, data and input hashes.

## Raw validation results

The validation split supplies 100 examples for each of seven current-static tasks. Three predeclared question forms rotate within each task; they do not triple the number of examples. Generation uses ordinary expert memory, batch size 8, at most 384 new tokens and seed 20261046. No independent test questions participate in this check.

| Measure | Correct / examples | Rate |
| --- | --- | --- |
| Parsed answer content | 272 / 700 | 38.86% |
| Canonical answer tag | 272 / 700 | 38.86% |
| Entire raw response, after trimming outer whitespace | 171 / 700 | 24.43% |
| Valid answer format | 700 / 700 | 100% |

The existing task parser compares the native answer fields; the canonical measure compares the extracted answer tags after whitespace normalization. Full-response matching also requires the surrounding generated text to match the expected response. These are different measures. The raw response is retained, and no engine or reference answer repairs it. Prose quality is not graded by this QA check.

| Task | Correct / 100 | Accuracy |
| --- | --- | --- |
| Piece counts (`counts`) | 53 | 53% |
| Diagonal query (`diagonal`) | 38 | 38% |
| File query (`file`) | 34 | 34% |
| Piece location (`locate`) | 20 | 20% |
| Material query (`materials`) | 30 | 30% |
| Piece identity (`piece`) | 71 | 71% |
| Rank query (`rank`) | 26 | 26% |

The separate token-level validation NLL was 0.1165288425. NLL is not an answer accuracy: a low average loss can coexist with incorrect answer fields, especially when much of an answer has a predictable format. The generated-answer results above are the measured course behavior.

## Continuation and interpretation

At 2,000 updates, global batch 256 accounts for 512,000 training-row presentations, about 16% of the first 3,200,005-row finite epoch. The course has not finished its first pass. Its first selection-eligible candidate is step 4000; the existing local minimum before plateau stopping is 20000 updates, with a maximum of 50000. These are the declared local run rules, not a claim that the paper used the same stopping rule.

The actual step-2000 selection decision is `stop=false` and `selected_step=null`. This early checkpoint is recorded, and training continues under the existing budget. The observed low answer accuracy is not evidence that the first course has succeeded. The 99% total and 95% per-task reference targets remain diagnostics rather than an independent stopping gate.

The resumption readback observed four live DDP ranks and contiguous finite updates through its recorded `observed_step`. The original first 2,000-update log prefix remains unchanged; subsequent batches continue the same epoch order. This verifies observed continuation, not a claim that all future steps or all courses have completed. Large immutable-input digest results are reused with file-signature checks; the reported reused byte count is hashing work avoided, not total training I/O.

All four courses must complete before full initial explanation SFT. Student-search distillation follows SFT and its development baseline. Strong play, useful human coaching, completed distillation and the [final independent acceptance](FINAL_INDEPENDENT_EVALUATION.md) remain unproven.

## Inspect the actual run

The actual course outputs remain under `runs/curriculum-paper-v5/static_current/`; the candidate, raw predictions and metrics are under `candidates/step-2000/`. The independent readback is `runs/paper-curriculum-v5-first-qa-readback-v1/`. Runtime datasets, model tensors and responses stay outside the Git source release.

Use the existing read-only observer from the repository root:

```bash
.venv/bin/python runs/verified-input-training-continuation-v3/observe_training.py --seconds 45
```

The owner session is `xqgeneral-paper-curriculum-preparation-v7`. The controller and exact process identities are recorded in its runtime continuation directory. The existing controller resumes the same configuration; do not launch another training owner or replace frozen execution code to update documentation.
