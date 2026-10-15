# Third formal curriculum validation

The fresh full-width run completed 6,000 optimizer updates and its third current-static raw validation on 2026-10-15 (UTC). On the same 700 questions, parsed answer content improved from 272/700 (38.86%) at step 2000, to 460/700 (65.71%) at step 4000, to 600/700 (85.71%) at step 6000. This remains an early first-course result; the four courses, explanation training, distillation and final playing-strength/coaching acceptance are unfinished.

The [independent readback](../evidence/paper-course-v5-third-raw-validation-v1.json) regenerated all 700 grading records from the retained raw responses and verified the predeclared questions, order and question variants. All 671,806,496 trainable FP32 parameters are finite and bitwise equal to corresponding tensors in the actual step-6000 source checkpoint. Updates 1–6000 are contiguous and finite; the original step-2000 and step-4000 log prefixes are preserved byte for byte. The [first](PAPER_FIRST_RAW_VALIDATION.md) and [second](PAPER_SECOND_RAW_VALIDATION.md) validation reports remain available.

## Results on the same questions

Generation retains ordinary expert memory, 100 examples per task, three question forms rotated within each task, batch size 8, at most 384 new tokens and seed 20261046. The three forms do not triple the 700 examples. Independent test questions do not participate.

| Measure | Step 2000 | Step 4000 | Step 6000 |
| --- | --- | --- | --- |
| Parsed answer content | 272/700 (38.86%) | 460/700 (65.71%) | 600/700 (85.71%) |
| Canonical answer tag | 272/700 (38.86%) | 460/700 (65.71%) | 600/700 (85.71%) |
| Entire raw response after trimming outer whitespace | 171/700 (24.43%) | 289/700 (41.29%) | 403/700 (57.57%) |
| Valid answer format | 700/700 (100%) | 700/700 (100%) | 700/700 (100%) |

Content grading compares parsed native answer fields. Canonical grading compares extracted answer tags after whitespace normalization. Entire-response matching also requires surrounding text to match the expected response; this QA check does not grade explanation prose. Raw mistakes remain available, without engine repair. Token-level NLL is a separate loss measure and cannot substitute for generated-answer accuracy.

| Task | Step 2000 correct / 100 | Step 4000 correct / 100 | Step 6000 correct / 100 | Change from 4000, percentage points |
| --- | --- | --- | --- | --- |
| Piece counts (`counts`) | 53 | 84 | 91 | +7 |
| Diagonal query (`diagonal`) | 38 | 41 | 87 | +46 |
| File query (`file`) | 34 | 82 | 96 | +14 |
| Piece location (`locate`) | 20 | 32 | 79 | +47 |
| Material query (`materials`) | 30 | 76 | 73 | -3 |
| Piece identity (`piece`) | 71 | 96 | 100 | +4 |
| Rank query (`rank`) | 26 | 49 | 74 | +25 |

The overall gain from step 4000 is 20.00 percentage points. The previously weak location, diagonal and rank tasks improved substantially. Material queries fell by three correct answers out of 100 and now set the minimum task accuracy at 73%; that decline remains visible. These measurements show continued learning on this validation set. They do not establish a convergence mechanism, guarantee later improvement, or demonstrate strong play or reliable coaching.

## Selection and actual continuation

The controller selected step 6000, recorded `stop=false`, zero checks without improvement and no stopping reason. Current-course raw accuracy selects the checkpoint; step 4000 remains the first eligible candidate. The first-course local minimum before plateau stopping remains 20,000 updates, with a maximum of 50,000. Overall 99% and per-task 95% reference targets remain diagnostic values.

At global batch 256, 6,000 updates represent 1,536,000 training-row presentations, approximately 48% of the first 3,200,005-row finite epoch. The first course has not completed its first pass. The run began from the pinned pretrained language student and a fresh full-width bridge; it did not inherit the earlier small-width curriculum.

The [resumption readback](../evidence/paper-course-v5-third-validation-resumption-v1.json) observed four actual live DDP ranks continuing from step 6000 through step 6026. Source, input hashes, compute contract, configuration and training-data metadata match the preceding segment. Captured updates are contiguous and finite, with resumed batches in the original epoch order. All 89 frozen GPU source modules were freshly hashed. This proves the recorded continuation, not completion of future training. Large unchanged input hashes are reused under the established file-identity checks.

All four courses must complete before full initial explanation SFT. Student-search distillation follows SFT and its development baseline. The [final independent evaluation](FINAL_INDEPENDENT_EVALUATION.md) remains pending.

## Inspect the run

The actual outputs are under `runs/curriculum-paper-v5/static_current/`; this candidate and its raw QA are under `candidates/step-6000/`. Independent grading and parameter checks are under `runs/paper-curriculum-v5-third-qa-readback-v1/`; actual resumption evidence is under `runs/paper-third-raw-validation-publication-v1/`. Runtime responses and model tensors stay outside the Git source release.

The existing read-only observer remains:

```bash
.venv/bin/python runs/verified-input-training-continuation-v3/observe_training.py --seconds 45
```

Training remains owned by `xqgeneral-paper-curriculum-preparation-v7`. Continue through the existing controller and frozen execution source.
