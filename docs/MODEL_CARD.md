# Xiangqi General model card

Xiangqi General couples a frozen Px0 Xiangqi Transformer expert with a Qwen3
language model through trainable gated cross-attention. It is an independent
adaptation of the [Queen paper](https://arxiv.org/abs/2610.03695v1).

The intended use is Chinese-chess study: propose a move, describe the board facts,
and explain verifiable continuations. Use the independent evaluation evidence to
decide how much to trust a checkpoint. A completed training run alone establishes
neither strong play nor accurate strategic explanations.

## Inputs and outputs

Input: Xiangqi FEN, complete move history, and a Chinese question. Coordinates
are `a0` to `i9`. Red is at the bottom. Structured move explanations include a
move, principal variation, candidates, evaluation perspective, and board facts.
Raw model generations must be evaluated before correction or oracle assistance.

## Training and evaluation

Pinned source/model identities are in `configs/sources.json`; per-experiment
configuration, hashes, code revisions, validation selection, and independent
test results are in run manifests. Synthetic legal games are split before QA
creation, and root/future positions are checked for overlap. Validation selects
checkpoints. Test games do not train or select checkpoints.

The original 80-step checkpoint is a pipeline pilot. It is not a strong-player
release. New trained models and measurements will be recorded in `STATUS.md`
after they have actually completed evaluation.

## Differences from the paper

This is Chinese chess, with Px0 rather than Lc0 and a pinned Qwen decoder. The
pilot uses four 384-wide bridge blocks rather than Queen's 16 large blocks.
The initial data is synthetic, and any local-model explanation teacher must be
named explicitly. The source code does not claim the paper's Elo, human-game
data coverage, paid teacher quality, or Western-chess rule equivalence.

## Limitations and release assets

AXF engine adjudication is a fixed software rule profile, not certification under
every competition rule book. Mechanical explanation checks cover declared
facts, moves, score perspectives, and continuations; strategic prose needs
separate human evaluation. A match stopped at a ply limit is reported as censored.

Source code uses GPL-3.0-or-later. Base weights and expert weights are downloaded
separately and excluded from Git. See `THIRD_PARTY_NOTICES.md` for asset licensing
and redistribution boundaries. This repository is not affiliated with the
Queen, Qwen, Px0, or Pikafish authors.
