# Xiangqi General development

- Work only in this project. Read `STATUS.md` for evidence and `README.md` for commands.
- Run `.venv/bin/python -m pytest -q` before committing a milestone. CPU tests require no model downloads.
- Keep models, datasets, checkpoints, vendor checkouts, and runtime logs out of Git.
- Preserve completed experiments; use a new output directory for changed configurations.
- Run training and batch generation in tmux. Record configuration, code revision, inputs, and output hashes.
- Split games before generating questions. Validation selects checkpoints; the independent test split never trains or selects models.
- Report raw model errors. Never repair an illegal predicted move with an oracle during evaluation.
- Keep the orchestration reusable; model loading belongs in `modeling.py`, rules in `rules.py`, and verification in separate modules.
- Commit coherent verified changes on a development branch. Merge validated milestones and push when a remote is available.
- Public-release facts belong in `docs/MODEL_CARD.md` and `THIRD_PARTY_NOTICES.md`; never publish unmeasured strength or blanket weight-license claims.
