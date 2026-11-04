# Xiangqi General model card

Xiangqi General is an independent Chinese-chess adaptation of the [Queen research pipeline](https://arxiv.org/abs/2610.03695v1), intended to play strong Xiangqi and explain moves in Chinese for personal study.

**No trained checkpoint has passed final playing-strength and coaching acceptance.** At the 2026-11-04 09:43 UTC readback, all four courses are complete and formal explanation SFT has started. The fourth course stopped at 48000 updates under the unchanged plateau rule and selected step 42000. Second- and third-course user-specified boundaries are preserved. Actual full-decoder four-GPU capacity and exact-resume checks have passed. Completing explanation SFT, sufficient real search distillation, independent acceptance and trained portable delivery remain outstanding.

The [actual full-model readback](FULL_DECODER_SFT_START.md) compares continuous four updates with two updates plus two restored updates for all 4,694,274,592 FP32 parameters, complete Adam and random states: no mismatched fields or tensors, with maximum parameter error zero. All 624 trainable tensors update. The capacity probe's peak Torch tensor allocation is 24.8100 GiB per GPU, including the restored leg at the same scale; this is neither total driver occupancy nor the peak of the complete formal run. The first three formal optimizer updates were read back. These facts prove capacity, restoration and actual training start, not chess strength or teaching quality.


The [portable inference directory](MODEL_INFERENCE_BUNDLE.md) is accepted by QA, explanation, match, single-move and planning evaluation. Each run binds every bundle asset and pins the resolved model directory before loading. Controlled CPU fixtures cover corruption, changes during generation, switched directory aliases, raw errors, completed-match resume and incomplete generation batches. Unpaired batch responses are preserved and receive no score. These checks do not evaluate the trained model or its playing strength.

## Current foundation evidence

| Scope | Latest step 48000 | Selected step 42000 |
| --- | ---: | ---: |
| Fourth course | 530 / 600 (88.33%) | 535 / 600 (89.17%) |
| Earlier three courses | 1920 / 2000 (96.00%) | 1924 / 2000 (96.20%) |
| Combined | 2450 / 2600 (94.23%) | 2459 / 2600 (94.58%) |

The [four-course completion readback](PAPER_FOUR_COURSES_COMPLETED.md) checks the actual complete curriculum, fourth-course manifest, normal controller exit, exact stopping policy and preserved user boundaries. It regrades 33800 archived answers across thirteen checkpoints with identical ordered development questions, variants and gold. The 2600 questions involve 66 declared game IDs, including 16 for the fourth course; these do not authenticate independent physical games. All latest formats are valid; prose is ungraded. The separate 26000-item independent test did not train or select the model.

Relative to 42k, the fourth-course, earlier-course and combined net changes are -5, -4 and -9 answers. Evasions score 80/100. The current-course rule retains 42k, with three checks without improvement after satisfying the 48k minimum. Both complete manifests and the controller's normal exit exist. Four-course completion does not establish broad playing strength or coaching quality.

The native proof of 48000 contiguous finite updates and all 671,806,496 finite source-equal FP32 trainables is reused. The selected 42k checkpoint bytes were freshly hashed. Original Adam arrays were not independently read back, and formal generated-token arrays were unavailable. Grading shares the frozen native implementation. No model inference or training was started by this publication readback.

Fixed step 36000 has now completed 400 evasion, 400 capture and 450 matched-depth answers, scoring 296/400 (74%), 283/400 (70.75%) and 312/450 (69.33%). Relative to step 26000, evasions improve by 16 answers while captures lose six. For the same 50 final positions, step 36000 scores 42/50 directly and 29/50 after eight halfmoves; 13 targets are correct directly but wrong at depth 8, with zero in the reverse direction. All nine depths are correct for 24/50 targets, versus 23/21/21 at steps 26000/32000/34000. This mixed result preserves a deep reasoning gap without identifying a unique training cause.

All 1250 new arrays and 2950 preserved reference arrays were decoded and regraded after the diagnostic GPU workers ended and the same original 38k QA physically resumed. A separate publication reader decoded all 4200 again. There are only 1200 unique question IDs: the 50 shared evasion/matched questions have identical generated text and final answers. Cohorts are reported separately, with no automatic model replacement or final test use. The step 32000 capture format error is preserved; all 1250 new answers have valid formats, EOS and no budget exhaustion. Decoder, native rules and grader are shared; prose and internal state are not independently certified.

The [earlier matched-depth control](PAPER_MATCHED_TARGET_DEPTH.md), [capture400 error audit](PAPER_CAPTURE400_ERROR_AUDIT.md), [full depth coverage](PAPER_FOURTH_COURSE_34000.md) and [queue admission](PAPER_STEP36000_DEVELOPMENT_REPLAY.md) retain their own inputs and historical scopes. Missing deep-row quantity alone was not supported by the complete 2,295,490-anchor coverage audit. The [earlier formal38k and diagnostic36k evidence](../evidence/paper-course4-38000-and-36000-development-v1.json) records the actual formal and diagnostic results without claiming a strong player or useful coach.

## Completed training diagnostics

The [four-arm short study and broader comparisons](PAPER_RATIONALE_LR_STUDY_RESULTS.md) are complete. Four same-parent, fresh-optimizer arms executed 632 actual updates in total; all 4000 original generated-token arrays were independently decoded. New-supervision arms score 462/600 and 424/600 on fourth-course validation, below the parent's 492/600. This short transfer result does not establish the outcome of full-scale new-target training.

The subsequent same-2600-question comparison decoded all 7800 original token arrays. The step-26000 control scores fourth-course 494/600, earlier-course 1928/2000 and combined 2422/2600; the short old-supervision/lower-learning-rate candidate scores 504/600, 1875/2000 and 2379/2600. On a separate 400-game-ID evasion cohort, the control scores 280/400 and the candidate 279/400. The candidate was not adopted. Its parent, optimizer and update budget differ from the continuously trained control, so this comparison does not isolate a long-run learning-rate effect. The common replay's older-question wording differs from the formal prompt set and yields one additional control answer; it must not be treated as the identical full prompt set used in every formal checkpoint.

The [actual step-24000 fit/memory diagnostic](PAPER_FOURTH_COURSE_FIT_DIAGNOSTIC.md) remains a bounded earlier diagnostic: original training samples score 526/600, normal validation 492/600, zero-memory validation 169/600 and shuffled-memory validation 133/600. All 2800 generated-token arrays were independently decoded. A separate evasion cohort scores 254/400, including 76/158 single-cannon checks. This establishes feature dependence and unresolved free-generation errors, without identifying a unique training cause.

The optional [future narration](PAPER_FORWARD_RATIONALE.md) and [evasion-mechanism profiles](PAPER_PARRY_RATIONALE.md) have CPU implementation checks and the short-study results above. Their [complete new-corpus production](PAPER_RATIONALE_CORPUS_PRODUCTION.md) has finished: 10,804,561 training, 2,600 validation and 26,000 test questions, with completed postwrite native split verification and byte bindings. Separate native source regeneration and all 10,833,161 baseline/profile row pairs have now completed, and the original job exited successfully. A second reader rehashed 111 frozen modules and rechecked 170 completed-receipt file identities; large input digests reuse the declared immutable-input receipts. At the 2026-11-02 03:15 UTC snapshot, the existing CPU consumer has completed 10,804,561 training and 2,600 validation indexes and entered full CPU preflight; token/feature checks remain unfinished, and no full-scale new-target run has been adopted. The base and incremental [source-history bundles](SOURCE_HISTORY.md) restore all 35 original commits through the auxiliary-evaluation fix and all 111 modules used by the native corpus experiment. [Worker-side text rendering](PAPER_RATIONALE_CPU_RENDERING.md) and [parallel split verification](PAPER_SPLIT_VERIFICATION.md) have bounded CPU timing controls. The current 1837-test suite is bound to 293 code, test, configuration and build files; these software results do not establish model gains. Active training and data jobs retain their own frozen sources.

## Remaining route and acceptance

The user-authorized [third-course step-26000 handoff](PAPER_FOURTH_COURSE_START_26000.md) remains the actual fourth-course parent. The source third-course run ended at 26564; its additional 564 updates are retained as discarded continuation and do not initialize this run. The [prefix contract](USER_AUTHORIZED_COURSE_PREFIX.md) records the earlier user selections.

All four actual courses are complete. A separate native CPU handoff is now complete: each of the four selected states has 226 shape-checked, finite FP32 tensors (671,806,496 parameters), and the exported 42k weights are byte-identical to their source. Native rule checks cover 2600 histories/answers and 2028 complete continuations. Large-cache digests reuse declared verified immutable-input receipts; the count of 4542 source bindings is not a claim to freshly read all cache bytes. No GPU network or SFT was started by this export. Real four-GPU full-decoder capacity/resume checks must still precede initial explanation SFT; the registered consumer will create its own same-source handoff inside its pipeline. [Engine-qualified explanations](ENGINE_QUALIFIED_EXPLANATIONS.md) retain 9823 unchanged training rows and byte-preserved 384 validation / 512 test rows; the [qualified successor](QUALIFIED_SFT_ROUTE.md) completed data/cache/token preflight and waits for available GPU capacity after the completed four-course parent. Full-label engine reference, teacher authorship and cross-review remain separately documented in [training-label validation](TRAINING_TEACHER_ENGINE_REFERENCE.md) and [prose provenance](REFERENCE_PROSE.md).

The [post-SFT curriculum check](POST_SFT_CURRICULUM_VALIDATION.md), raw explanation validation and development matches still require the actually trained model. Sufficient real student-search yield and distillation rounds follow measured results; the 512-root yield pilot does not satisfy formal distillation or final acceptance.

The [final independent protocol](FINAL_INDEPENDENT_EVALUATION.md) prepares 26000 QA items, 512 explanations and 384 matches from 64 protected test games. No final checkpoint or settings have been locked and scored. Those inputs do not establish playing strength or a useful coach. [Current foundation evidence](../evidence/paper-four-courses-completed-v1.json), [matched-depth result evidence](../evidence/paper-matched-target-depth-results-v1.json), [new-corpus production evidence](../evidence/paper-rationale-corpus-production-v1.json), [STATUS.md](../STATUS.md) and [historical snapshots](MODEL_CARD_HISTORY.md) distinguish current claims, prepared inputs and preserved earlier milestones.

## Intended use and interface

An opt-in [microbatch padding crop](MICROBATCH_PADDING.md) preserves the legacy default. On the same 2,048 sampled training rows, the actual helper preserved valid inputs and supervised targets while reducing padded token positions by 28.14%. Tiny randomly initialized Qwen3 models with real DenseBridge blocks passed CPU output, loss, gradient and exact continuation controls. These are CPU implementation checks and token counts: full 4B/BF16 GPU equivalence, throughput and model-quality improvements from that crop remain unmeasured. The completed curriculum retained its original 89-module GPU snapshot. The earlier authorized 108-module SFT snapshot is preserved; actual formal SFT uses its separately tested 109-module parameter-sharding successor, with the original padding default and finite traversal.

The intended use is studying positions, comparing candidate moves, explaining verifiable continuations and reviewing games. A completed training run or high curriculum accuracy alone does not establish useful playing strength or good coaching.

The terminal study interface resumes the complete saved history and writes new moves, undo operations and raw advice back to that record by default. An explicit `--output` saves a separate branch while preserving the input file. Command-line regression checks cover both paths; they use a controlled predictor and do not establish trained-model quality. See the [usage entry](../README.md#原型推理) and [preserved development guide](DEVELOPMENT_GUIDE.md).

Input is a Xiangqi FEN, the supplied move history and a Chinese question. Coordinates run from `a0` to `i9`, with Red at the bottom. Tactical fragments explicitly retain missing earlier-history boundaries. Explanation output includes a recommended move, principal variation, candidates, numerical evaluation and board facts; raw text, malformed answers and illegal moves remain available for evaluation.

## Models and architecture

| Component | Current formal plan |
| --- | --- |
| Language student | Qwen3-4B-Instruct-2507, pinned pretrained revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Chess expert | Frozen Px0 Xiangqi Transformer; network identity and feature layers are bound in each run |
| Bridge | 16 full-width gated cross-attention blocks, width 2560, 16 heads, two-times-width ReLU-squared feed-forward layers and zero-initialized gates |
| Board input | 105 learned board tokens and 90 expert square features; the formal latent curriculum supplies no literal board-text dictionary |
| Curriculum parameters | 671,806,496 trainable FP32 bridge/board parameters with BF16 computation; the pretrained language decoder and expert are frozen |
| Explanation SFT | Full decoder plus bridge/board parameters; 4,694,274,592 trainable parameters in the configured model, using FP32 master values and BF16 computation |
| Initial prose teacher | User-authorized GPT-6-Astra Low annotation and review |
| Search consolidator | Official unquantized Qwen3.8-27B weights with BF16 inference; 27,356,728,560 active model parameters, with 424,699,392 inactive MTP parameters preserved in the checkpoint |

The formal curriculum starts from the pinned pretrained student and a fresh bridge. It does not inherit the earlier small-width curriculum or explanation checkpoints. Per-run manifests bind the actual copied configuration, code and model files; [source identities](../configs/sources.json) and the [third-party notices](../THIRD_PARTY_NOTICES.md) describe upstream assets.

## Data and current preparation

The completed counts and audits below describe the original formal curriculum. The separate extended-rationale corpus remains subject to its own complete source pairing and token/feature preflight.

The regenerated 26-task curriculum has **10,804,561 training, 2,600 validation and 26,000 independent-test questions**. Games are split before question generation, and current/future board footprints are checked for cross-split overlap. [Source-based native reconstruction, answer checks and byte comparison](../evidence/paper-complete-native-readback-v1.json) have completed for all 10,833,161 questions. This proves those mechanical data contracts; it does not authenticate participant identities, establish optimal recorded moves or measure a trained student's abilities.

All 10,807,161 training/validation examples and 5,200 additional validation-format encodings have passed the token stage. The largest checked training/validation example is 327 tokens. The [full preflight](../evidence/paper-course-full-preflight-complete-v1.json) has also completed 2,081,078 unique-history replays, complete split-isolation checks and hashes for 4,143 bound input files totaling 3,386,878,285,696 bytes. The [independent source/cache/base readback](../evidence/paper-course-full-source-readback-complete-v1.json) has completed, and the formal continuous four-GPU capacity gate passed before training began. Independent test answers were not included in the token stage or training/selection budget.

The original reviewed initial explanation corpus has **10,758 training, 384 validation and 512 independent-test rows**. The active qualified successor retains **9,823 training rows**, with the same byte-identical validation and independent-test files. Existing color-derived rows preserve their parent split and do not count as independent games or teacher calls. [Reference-prose provenance](REFERENCE_PROSE.md) records authoring, cross-review and corrections. Recorded-game categories and names are source declarations; native legal replay does not establish human identity or unassisted play. Raw and derived datasets are excluded from the source release.

An additional [anonymous PvP source](PUBLIC_ANONYMOUS_GAMES.md) provides 8,828 independently read-back native full histories (486,754 plies), split into 7,072 train / 886 validation / 870 test games before question generation. Source PvP labels do not authenticate human identity or unassisted play. Source analysis and CPL are not neural labels. The [expanded unused-history pool](PUBLIC_RECORDED_SEARCH_SOURCES.md#扩充候选池-v3) now contains 229,882 candidate histories from 45,268 canonical training records, including games, fragments and isolated positions. The new anonymous source contributes 42,432 histories. All candidates were compared field by field, all ten canonical files were re-streamed and hashed, and 3,366,490 reserved root/future/color positions were reconstructed. Actual student-search yield and model benefit remain unmeasured; the current course, initial SFT and v2-pilot inputs retain their frozen versions. These are search inputs, with zero new neural labels.

## Training plan

The [formal curriculum recipe](PAPER_FOUNDATION_EXPERIMENT.md) uses finite mixed-data traversal, answer-token-average loss, four GPUs, global batch 256 and microbatch 4. Old-course examples participate in training as well as separate replay diagnostics.

| Course | Training mixture in course order | Maximum updates | Local minimum before plateau stopping |
| --- | --- | --- | --- |
| Current static | 100% | 50,000 | 20,000 |
| Current dynamic | 10% / 90% | 60,000 | 32,000 |
| Future static | 6% / 8% / 86% | 30,000 | 22,000 |
| Future dynamic | 6% / 6% / 8% / 80% | 100,000 | 48,000 |

Current-course raw validation accuracy selects checkpoints; earlier-course replay is reported separately. Earliest ties, eligibility after warmup, three checks without improvement and the minimum stopping budgets are explicit local rules. The paper and released configuration do not establish a complete final selection rule. Local total/per-task target accuracies are diagnostics and do not independently stop this run; the budgets above are plans, not completed update counts.

[Initial explanation SFT](PAPER_POST_CURRICULUM.md) follows the actual completed new four-course checkpoint. The [qualified-data successor](QUALIFIED_SFT_ROUTE.md) now uses 9,823 retained training labels; four finite passes compile to 2,456 optimizer updates at global batch 16 and microbatch 1. The earlier 2,692-update, 10,758-label queue was retired with its original inputs and logs preserved. Validation NLL selects the SFT checkpoint; raw move and prose checks then assess the same model. The actual full-decoder capacity and restoration control has passed, and formal SFT has started. Full-BF16 teacher multi-GPU settings are prepared separately; their actual inference capacity is not established by CPU protocol checks.

[Student-search generation](PAPER_SEARCH_GENERATION.md) uses the trained student for numerical analysis and recursive critical-line searches, followed by the full BF16 consolidator. The 512-root pilot measures usable-label yield before any expansion; it is not a finished distillation dataset or the final delivery scope. Search volume and subsequent training rounds must be chosen from actual yield, development matches and explanation quality. [Distillation training and development evaluation](PAPER_DISTILLATION_TRAINING.md) retain the selected model and its ancestry.

## Evaluation and known limits

Validation selects checkpoints; independent test games and labels do not train or select them. Development evaluation compares raw answers with normal, zeroed and mismatched expert memory, checks native facts and variations, and uses blinded auxiliary prose ratings. The prose reviewer shares its base model with the consolidator, so those scores are not independent human judgments.

The planned SFT development match baseline covers 64 protected validation openings, both colors and Pikafish budgets of 100, 1,000 and 10,000 nodes: 384 games, with at most 256 additional plies each. Matches preserve raw turns and interruptions; ply-limited games are censored. Illegal raw moves or variations count as errors and are never repaired with an oracle in raw evaluation. The optional legal-move-trie decoder is reported separately because its output legality is imposed by construction.

AXF adjudication is a fixed software rule profile. Mechanical explanation checks verify declared facts, moves, score perspectives and continuations; they do not establish that an explanation teaches a human well. Final independent playing-strength and coaching evaluation, including analysis of failure cases, remains pending. No paper Elo, useful-coach claim or improvement from completed distillation is reported for the new run.

Earlier pretrained GPU controls verify frozen-decoder updates and portable-inference equivalence within their stated fixture budgets. The new actual full-model control verifies complete parameters, Adam and exact interruption/restoration; formal explanation training has started. Neither those controls nor CPU software tests establish completed SFT, chess strength or coaching quality. Earlier measured checkpoint failures and their limits remain in the historical record and linked evidence.

## Delivery and distribution

The [portable inference bundle](MODEL_INFERENCE_BUNDLE.md) carries the decoder, tokenizer, all trained tensors and matching expert assets, with per-file hashes and runtime configuration. Its completed relocation control used a two-update frozen-decoder capacity checkpoint; a formally trained and independently accepted delivery bundle has not been produced.

The [109-module execution-source archive](../reproducibility/full-decoder-fsdp2-v1-109-source.tar.gz) preserves the actual full-model probe and initial-SFT implementation, separate from the evolving repository head. Every module hash and a real isolated import after extraction match the native execution identity. It includes source and notices, without models, datasets, logs, credentials or test inputs; it does not replace trained-model delivery or the original data and resume contracts.

Project source is GPL-3.0-or-later. Base-model notices and separate expert/NNUE distribution rights apply to weights; the code license does not establish permission to redistribute those assets. Models, datasets, feature caches and runtime logs are excluded from the Git source release. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the existing asset-specific boundaries.
