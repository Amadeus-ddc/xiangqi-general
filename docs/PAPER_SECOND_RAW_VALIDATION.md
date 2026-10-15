# Second formal curriculum validation

The fresh full-width run completed 4,000 optimizer updates and its second raw current-static validation on 2026-10-15. The same 700 validation questions scored 460/700 (65.71%) on parsed answer content, compared with 272/700 (38.86%) at step 2000. This is an early first-course checkpoint; the four courses, explanation training, distillation and final playing-strength/coaching acceptance remain unfinished.

The [independent readback](../evidence/paper-course-v5-second-raw-validation-v1.json) regenerated every grading record from the retained raw model response and verified all declared questions, their order and question variants. All 671,806,496 trainable FP32 parameters are finite and bitwise equal to the corresponding tensors in the actual step-4000 source checkpoint. The first 4,000 updates are contiguous and finite, and the original 2,000-update log prefix is preserved byte for byte. The [first validation report](PAPER_FIRST_RAW_VALIDATION.md) remains available.

## Results on the same validation questions

Generation retains ordinary expert memory, 100 examples per task, three question forms rotated within each task, batch size 8, at most 384 new tokens and seed 20261046. The three forms do not triple the 700 examples. No independent test questions participate.

| Measure | Step 2000 | Step 4000 | Change in percentage points |
| --- | --- | --- | --- |
| Parsed answer content | 272/700 (38.86%) | 460/700 (65.71%) | +26.86 |
| Canonical answer tag | 272/700 (38.86%) | 460/700 (65.71%) | +26.86 |
| Entire raw response after trimming outer whitespace | 171/700 (24.43%) | 289/700 (41.29%) | +16.86 |
| Valid answer format | 700/700 (100%) | 700/700 (100%) | 0 |

The content metric compares parsed native answer fields. The canonical metric compares extracted answer tags after whitespace normalization. Entire-response matching also requires the surrounding text to match the expected response. Raw answers and mistakes are retained, with no engine repair. This QA check does not grade explanation prose.

| Task | Step 2000 correct / 100 | Step 4000 correct / 100 | Change in percentage points |
| --- | --- | --- | --- |
| Piece counts (`counts`) | 53 | 84 | +31 |
| Diagonal query (`diagonal`) | 38 | 41 | +3 |
| File query (`file`) | 34 | 82 | +48 |
| Piece location (`locate`) | 20 | 32 | +12 |
| Material query (`materials`) | 30 | 76 | +46 |
| Piece identity (`piece`) | 71 | 96 | +25 |
| Rank query (`rank`) | 26 | 49 | +23 |

All seven scores increased on this fixed validation set. Location, diagonals and rank queries remain the weakest tasks. Repeated validation results support checkpoint selection; they are not independent-test scores or evidence of strong play or reliable coaching. Token-level NLL is a separate loss measure and must not be substituted for these generated-answer scores.

## Selection and observed continuation

Step 4000 is the first selection-eligible candidate. The actual controller selected step 4000 with accuracy 65.71% and minimum task accuracy 32%, while recording `stop=false`, zero checks without improvement and no stopping reason. The first-course local minimum before plateau stopping remains 20,000 updates, with a maximum of 50,000. The 99% overall and 95% per-task reference targets remain diagnostic values.

At global batch 256, 4,000 updates account for 1,024,000 training-row presentations, approximately 32% of the first 3,200,005-row finite epoch. The first course has not completed its first pass. Current-course raw validation accuracy selects checkpoints; old-course training mixtures and later replay diagnostics retain the declared curriculum recipe.

The [resumption readback](../evidence/paper-course-v5-second-validation-resumption-v1.json) observed four actual live DDP ranks continuing from step 4000 through step 4021. Source, input hashes, compute contract, configuration and training-data metadata match the preceding segment. All captured updates remain contiguous and finite, and resumed batches follow the original epoch order. All 89 frozen GPU source modules were freshly hashed. This proves the recorded continuation, not completion of future training. Large unchanged input hashes are reused under the existing file-identity checks.

All four courses must complete before full initial explanation SFT. Student-search distillation follows SFT and its development baseline. The [final independent evaluation](FINAL_INDEPENDENT_EVALUATION.md) remains pending.

## Inspect the run

The actual outputs are under `runs/curriculum-paper-v5/static_current/`; this candidate and its raw QA are under `candidates/step-4000/`. Independent grading and tensor checks are under `runs/paper-curriculum-v5-second-qa-readback-v1/`; observed continuation is under `runs/paper-second-raw-validation-publication-v1/`. Runtime responses and model tensors stay outside the Git source release.

The existing read-only observer remains:

```bash
.venv/bin/python runs/verified-input-training-continuation-v3/observe_training.py --seconds 45
```

The training owner remains `xqgeneral-paper-curriculum-preparation-v7`. Continue through the existing controller and frozen execution source.
