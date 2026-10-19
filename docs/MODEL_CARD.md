# Xiangqi General model card

Xiangqi General is an independent Chinese-chess adaptation of the [Queen research pipeline](https://arxiv.org/abs/2610.03695v1). Its goal is a model that plays strong Xiangqi and explains moves in Chinese for personal study.

**No trained checkpoint has passed final playing-strength and coaching acceptance.** The second course is training. [Step-10000 validation](PAPER_SECOND_COURSE_10000.md) scored 540/600 on current dynamic questions (90.00%, previously 89.17%) and 663/700 on static replay (94.71%, previously 93.57%). The same 1,300 ordered variants were compared; current-course gains/losses were 29/24 and replay gains/losses 28/20. Control reached 76%, with exact nonempty answers unchanged at 26/48. Checking-move enumeration declined from 97 to 92/100; replay piece counts declined from 100 to 97/100. Five completed full CPU readbacks were reused. Live step-10000 QA and CPU identities were captured; the historical step-2000 GPU binding remains unavailable. The unchanged comparison reader completed using Python safe-path mode after a local standard-library import collision; the original failure is preserved. Step 10000 is selected by the frozen current-course metric, separately from the 92.54% mixed total. Original four-rank continuation was observed from 10066 to 10071 and its trace captured through 10085; next QA is 12000, with a 32000/60000 stopping minimum/maximum. The first complete finite epoch contained 2,343,401 valid rows and seven padding rows. Through step 10000, 2,559,977 valid presentations were verified; nominal step-times-batch capacity is reported separately. The first course completed at 30000 and selected step 24000 (682/700, 97.43%). One of four courses is complete. Full initial explanation SFT, actual student-search distillation and independent playing/coaching acceptance remain incomplete. See [STATUS.md](../STATUS.md) and [historical snapshots](MODEL_CARD_HISTORY.md).

The [final independent protocol](FINAL_INDEPENDENT_EVALUATION.md) fixes 26,000 test QA items, 512 test explanations and 384 planned matches from 64 distinct test games. [Input preparation](../evidence/final-independent-evaluation-inputs-v1.json) checked the protected files and replayed the selected opening prefixes, with no selected-game overlap with current training or validation. No final checkpoint has been locked or scored; these are prepared inputs rather than model-quality results.

## Intended use and interface

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

The reviewed initial explanation corpus has **10,758 training, 384 validation and 512 independent-test rows**. Existing color-derived rows preserve their parent split and do not count as independent games or teacher calls. [Reference-prose provenance](REFERENCE_PROSE.md) records authoring, cross-review and corrections. Recorded-game categories and names are source declarations; native legal replay does not establish human identity or unassisted play. Raw and derived datasets are excluded from the source release.

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

[Initial explanation SFT](PAPER_POST_CURRICULUM.md) follows the actual completed new four-course checkpoint. Four finite passes compile to 2,692 optimizer updates at global batch 16 and microbatch 1. Validation NLL selects the SFT checkpoint; raw move and prose checks then assess the same model. Full-decoder GPU capacity and restoration still require their actual post-course control.

[Student-search generation](PAPER_SEARCH_GENERATION.md) uses the trained student for numerical analysis and recursive critical-line searches, followed by the full BF16 consolidator. The 512-root pilot measures usable-label yield before any expansion; it is not a finished distillation dataset or the final delivery scope. Search volume and subsequent training rounds must be chosen from actual yield, development matches and explanation quality. [Distillation training and development evaluation](PAPER_DISTILLATION_TRAINING.md) retain the selected model and its ancestry.

## Evaluation and known limits

Validation selects checkpoints; independent test games and labels do not train or select them. Development evaluation compares raw answers with normal, zeroed and mismatched expert memory, checks native facts and variations, and uses blinded auxiliary prose ratings. The prose reviewer shares its base model with the consolidator, so those scores are not independent human judgments.

The planned SFT development match baseline covers 64 protected validation openings, both colors and Pikafish budgets of 100, 1,000 and 10,000 nodes: 384 games, with at most 256 additional plies each. Matches preserve raw turns and interruptions; ply-limited games are censored. Illegal raw moves or variations count as errors and are never repaired with an oracle in raw evaluation. The optional legal-move-trie decoder is reported separately because its output legality is imposed by construction.

AXF adjudication is a fixed software rule profile. Mechanical explanation checks verify declared facts, moves, score perspectives and continuations; they do not establish that an explanation teaches a human well. Final independent playing-strength and coaching evaluation, including analysis of failure cases, remains pending. No paper Elo, useful-coach claim or improvement from completed distillation is reported for the new run.

Real pretrained GPU controls verify frozen-decoder curriculum updates, interruption/restoration and portable-inference equivalence. These are bounded fixtures, not completed courses or full-decoder SFT. CPU tests verify software contracts, not chess strength. Earlier measured checkpoint failures, control budgets and their limits remain in the historical record and linked evidence.

## Delivery and distribution

The [portable inference bundle](MODEL_INFERENCE_BUNDLE.md) carries the decoder, tokenizer, all trained tensors and matching expert assets, with per-file hashes and runtime configuration. Its completed relocation control used a two-update frozen-decoder capacity checkpoint; a formally trained and independently accepted delivery bundle has not been produced.

Project source is GPL-3.0-or-later. Base-model notices and separate expert/NNUE distribution rights apply to weights; the code license does not establish permission to redistribute those assets. Models, datasets, feature caches and runtime logs are excluded from the Git source release. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the existing asset-specific boundaries.
