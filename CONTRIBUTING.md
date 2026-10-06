# Contributing

Use Python 3.11 or newer. Install a suitable PyTorch build, then `pip install -e '.[dev]'`.
Run `python -m pytest -q` before opening a pull request. These tests use synthetic
fixtures and run on CPU without downloading weights.

For research changes, include the configuration, seed, data split, verification,
and limitations. Store large generated assets outside Git and include hashes in
the evidence manifest. Do not overwrite completed experiments or evaluate on
training games. Claims about playing strength need independent game results.

All contributions use GPL-3.0-or-later. Keep upstream notices intact. Do not
include credentials, machine-specific configuration, or model weights in a PR.
