# Xiangqi General model card

Xiangqi General is an independent Chinese-chess adaptation of the [Queen research pipeline](https://arxiv.org/abs/2610.03695v1). Its goal is a model that plays strong Xiangqi and explains moves in Chinese for personal study.

**No trained checkpoint has passed final playing-strength and coaching acceptance.** The [course-three step-4000 QA](PAPER_THIRD_COURSE_4000.md) and same-prompt comparison were independently verified on 2026-10-23 at 05:50 UTC+8: 630/700 current questions (90.00%, minimum task 75%) and 1214/1300 earlier-course questions (93.38%). All 2000 ordered questions, original scores and 671806496 actual FP32 parameters passed CPU readback. All response formats were valid; wrong answers remain errors. The original four-rank continuation through updates 4001–4008 was separately verified. The native selector chose step 4000, with the next QA at 6000. Two of four courses are complete; explanation SFT, sufficient real search/distillation, independent acceptance and trained inference delivery remain pending.

Both course-three rounds used exactly the same 2000 IDs, order, prompts, variants and gold answers. The current course rose from 625 to 630 correct (41 fixes, 36 new errors); earlier courses fell from 1223 to 1214 (41 fixes, 50 new errors). The comparison regraded all 4000 raw answers with the unchanged native scorer. There is no measured zero-step course-three baseline, and two QA points do not establish convergence or final playing strength. The separate [first-round comparison](PAPER_THIRD_COURSE_2000.md) with the second-course parent still retains its 878 changed variants and 422 matching prompts; those cross-course limits do not replace this full within-course pairing.

The user explicitly selected the second-course step-30000 parent (584/600 current, 668/700 replay); its original 32000-update minimum remains unmet. The [explicit-prefix handoff](USER_AUTHORIZED_COURSE_PREFIX.md) requires the exact caller-bound authorization and all four actual courses. Its frozen 108-module successor owns the waiting queue; original 89-module GPU training and the postcourse recipe remain unchanged. Full native four-course export and actual full-decoder capacity/resume proof remain required before SFT. Actual checkpoint readback is distinct from the earlier two-course native/header preflight; large input hashes were reused, and the original optimizer arrays were not independently read.

[Full training-label engine reference](TRAINING_TEACHER_ENGINE_REFERENCE.md) independently read back all 10,758 original labels: 10,644 recommended first moves and 9,823 complete variations passed the declared loss threshold. [Qualified data](ENGINE_QUALIFIED_EXPLANATIONS.md) retain 9,823 unchanged training records with 384 validation and 512 test records byte-preserved. Native indexed loading and tokenization passed, with maximum lengths 649/625 under 1,024 tokens. The [qualified successor](QUALIFIED_SFT_ROUTE.md) completed its actual data/cache/token preflight, owned queue handoff and the separate published 1,630-test milestone. Actual four-GPU full-decoder capacity and resume checks still follow all four completed courses.

The [native reference generator](TRAINING_LABEL_REFERENCE_GENERATOR.md) now provides reproducible scoring, retained requests, a separate reader process and explicit resume. Its real capability run covered eight original training labels, 46 main-variation plies and 58 requests, including one same-best additional score. This does not repeat the existing full-corpus experiment or measure student quality. Full explanation SFT, actual student-search distillation and independent playing/coaching acceptance remain incomplete. See [STATUS.md](../STATUS.md) and [historical snapshots](MODEL_CARD_HISTORY.md).

The [final independent protocol](FINAL_INDEPENDENT_EVALUATION.md) fixes 26,000 test QA items, 512 test explanations and 384 planned matches from 64 distinct test games. [Input preparation](../evidence/final-independent-evaluation-inputs-v1.json) checked the protected files and replayed the selected opening prefixes, with no selected-game overlap with current training or validation. No final checkpoint has been locked or scored; these are prepared inputs rather than model-quality results.

## Intended use and interface

An opt-in [microbatch padding crop](MICROBATCH_PADDING.md) preserves the legacy default. On the same 2,048 sampled training rows, the actual helper preserved valid inputs and supervised targets while reducing padded token positions by 28.14%. Tiny randomly initialized Qwen3 models with real DenseBridge blocks passed CPU output, loss, gradient and exact continuation controls. These are CPU implementation checks and token counts: full 4B/BF16 GPU equivalence, throughput and model-quality improvements remain unmeasured. The curriculum retains its original 89-module GPU snapshot. The authorized 108-module SFT successor is separately frozen; its recipe keeps the existing padding default and finite data traversal.

The intended use is studying positions, comparing candidate moves, explaining verifiable continuations and reviewing games. A completed training run or high curriculum accuracy alone does not establish useful playing strength or good coaching.

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

[Initial explanation SFT](PAPER_POST_CURRICULUM.md) follows the actual completed new four-course checkpoint. The [qualified-data successor](QUALIFIED_SFT_ROUTE.md) now uses 9,823 retained training labels; four finite passes compile to 2,456 optimizer updates at global batch 16 and microbatch 1. The earlier 2,692-update, 10,758-label queue was retired with its original inputs and logs preserved. Validation NLL selects the SFT checkpoint; raw move and prose checks then assess the same model. Full-decoder GPU capacity and restoration still require their actual post-course control.

[Student-search generation](PAPER_SEARCH_GENERATION.md) uses the trained student for numerical analysis and recursive critical-line searches, followed by the full BF16 consolidator. The 512-root pilot measures usable-label yield before any expansion; it is not a finished distillation dataset or the final delivery scope. Search volume and subsequent training rounds must be chosen from actual yield, development matches and explanation quality. [Distillation training and development evaluation](PAPER_DISTILLATION_TRAINING.md) retain the selected model and its ancestry.

## Evaluation and known limits

Validation selects checkpoints; independent test games and labels do not train or select them. Development evaluation compares raw answers with normal, zeroed and mismatched expert memory, checks native facts and variations, and uses blinded auxiliary prose ratings. The prose reviewer shares its base model with the consolidator, so those scores are not independent human judgments.

The planned SFT development match baseline covers 64 protected validation openings, both colors and Pikafish budgets of 100, 1,000 and 10,000 nodes: 384 games, with at most 256 additional plies each. Matches preserve raw turns and interruptions; ply-limited games are censored. Illegal raw moves or variations count as errors and are never repaired with an oracle in raw evaluation. The optional legal-move-trie decoder is reported separately because its output legality is imposed by construction.

AXF adjudication is a fixed software rule profile. Mechanical explanation checks verify declared facts, moves, score perspectives and continuations; they do not establish that an explanation teaches a human well. Final independent playing-strength and coaching evaluation, including analysis of failure cases, remains pending. No paper Elo, useful-coach claim or improvement from completed distillation is reported for the new run.

Real pretrained GPU controls verify frozen-decoder curriculum updates, interruption/restoration and portable-inference equivalence. These are bounded fixtures, not completed courses or full-decoder SFT. CPU tests verify software contracts, not chess strength. Earlier measured checkpoint failures, control budgets and their limits remain in the historical record and linked evidence.

## Delivery and distribution

The [portable inference bundle](MODEL_INFERENCE_BUNDLE.md) carries the decoder, tokenizer, all trained tensors and matching expert assets, with per-file hashes and runtime configuration. Its completed relocation control used a two-update frozen-decoder capacity checkpoint; a formally trained and independently accepted delivery bundle has not been produced.

Project source is GPL-3.0-or-later. Base-model notices and separate expert/NNUE distribution rights apply to weights; the code license does not establish permission to redistribute those assets. Models, datasets, feature caches and runtime logs are excluded from the Git source release. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the existing asset-specific boundaries.
