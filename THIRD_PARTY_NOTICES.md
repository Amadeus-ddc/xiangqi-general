# Third-party sources and model assets

Project code is GPL-3.0-or-later. `expert.py` and the pinned network descriptor
derive from Px0's GPL-3.0-or-later network mathematics and protocol. The native
reference calls the unmodified upstream exporter and encoder. Preserve upstream
copyright notices when distributing these sources.

| Component | Source | License / distribution boundary |
| --- | --- | --- |
| Queen research and architecture | https://github.com/queen-project/queen | Apache-2.0; implementation reference, upstream checkout excluded from releases |
| Px0 and network protocol | https://github.com/official-pikafish/px0 | GPL-3.0-or-later; see `LICENSE` and pinned `configs/sources.json` |
| Pikafish labeling engine | https://github.com/official-pikafish/Pikafish | GPL-3.0-or-later; downloaded/built separately |
| abseil-cpp native headers | https://github.com/abseil/abseil-cpp | Apache-2.0; separate upstream checkout |
| Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 in the pinned asset; downloaded separately |
| Px0 network weights | https://github.com/official-pikafish/pxzero-networks | Weight redistribution permission is not established by the code license; not bundled |
| Pikafish NNUE weights | https://github.com/official-pikafish/Pikafish | Obtain via the upstream engine; not bundled |
| pyffish | https://github.com/fairy-stockfish/Fairy-Stockfish | GPL-3.0; pinned to 0.0.90 |

This project is an independent Chinese-chess adaptation, not an official Queen
release or reproduction of the published Western-chess checkpoint. Checkpoint
distribution must retain the base-model notices and separately establish rights
for any included expert or NNUE assets. The source release contains no weights.
