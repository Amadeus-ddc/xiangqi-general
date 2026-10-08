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
| Qwen3.8-27B consolidation teacher | https://huggingface.co/Qwen/Qwen3.8-27B | Apache-2.0 in the [pinned official license](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE); full BF16 inference only, downloaded separately |
| Px0 network weights | https://github.com/official-pikafish/pxzero-networks | Weight redistribution permission is not established by the code license; not bundled |
| Pikafish NNUE weights | https://github.com/official-pikafish/Pikafish | Obtain via the upstream engine; not bundled |
| pyffish | https://github.com/fairy-stockfish/Fairy-Stockfish | GPL-3.0; pinned to 0.0.90 |
| CCPD recorded games and tactical lines | https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset | Source declares CC-BY-4.0 at pinned revision `368a47a947773dd8692c026e286dd19b6277b993`; preserve attribution and local `SOURCE_LICENSE`; raw and derived records are excluded from the code release |
| Recent public recorded games | https://www.xiangqiqipu.com | Source license/redistribution permission is unspecified; local factual main lines and source hashes only, no site prose or analysis branches used as labels; records and pages are not bundled |
| Additional bundled PGN collections | https://github.com/CGLemon/chinese-chess-PGN, mirrored at https://github.com/v-azhu/alphazetacchess/tree/0d550401ea9b0c8542e01843ce03a708326692ff/trainingdata | WXF and Dongping collections as described by the source; mirror Git blob identities and actual bytes verified. Dataset license/redistribution permission is unspecified; raw archives, records and derived datasets are excluded from the code release, and source identity/category assertions remain unauthenticated |
| Public PlayStrategy game exports | https://playstrategy.org/api | Original public NDJSON/PGN responses captured without credentials; source license/redistribution permission is unspecified. Full native histories and exact export parity are checked; user/BOT labels do not authenticate real identities or unassisted play. Original exports, derived game datasets and site text are excluded from the source release |

This project is an independent Chinese-chess adaptation, not an official Queen
release or reproduction of the published Western-chess checkpoint. Checkpoint
distribution must retain the base-model notices and separately establish rights
for any included expert or NNUE assets. The source release contains no weights.

Recorded-game categories and participant names are source assertions, not independent
authenticity or identity verification. Native legal replay does not establish move
optimality or rights to publish a trained checkpoint. Dataset notices do not change
the separate base, expert and checkpoint distribution review.
