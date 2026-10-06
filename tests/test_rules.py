import pytest
from xqgeneral.rules import START_FEN, legal_moves, play, piece_map, from_fairy, to_fairy, replay
from xqgeneral.data import make_records
from xqgeneral.expert import encode_history


def test_coordinate_roundtrip_and_start_position():
    for rank in range(10):
        move = f"a{rank}i{9-rank}"
        assert from_fairy(to_fairy(move)) == move
    assert "b2b9" in legal_moves(START_FEN)
    assert len(legal_moves(START_FEN)) == 44
    board = piece_map(START_FEN)
    assert (board["b0"], board["e0"], board["e9"]) == ("H", "K", "k")
    after = play(START_FEN, "b0c2")
    assert "b0" not in piece_map(after)
    assert piece_map(after)["c2"] == "H"


def test_horse_leg_and_elephant_eye():
    blocked_horse = START_FEN.replace("/9/RNBAKABNR", "/1P7/RNBAKABNR")
    assert "b0c2" in legal_moves(START_FEN)
    assert "b0c2" not in legal_moves(blocked_horse)
    blocked_elephant = START_FEN.replace("/9/RNBAKABNR", "/3P5/RNBAKABNR")
    assert "c0e2" in legal_moves(START_FEN)
    assert "c0e2" not in legal_moves(blocked_elephant)


def test_cannon_screen_and_flying_generals():
    assert "b2b9" in legal_moves(START_FEN)
    assert "b2b7" not in legal_moves(START_FEN)
    pinned = "4k4/9/9/9/9/9/9/9/4R4/4K4 w - - 0 1"
    assert "e1d1" not in legal_moves(pinned)
    assert "e1e2" in legal_moves(pinned)


def test_pawn_and_illegal_replay():
    assert "a3b3" not in legal_moves(START_FEN)
    assert "a3a4" in legal_moves(START_FEN)
    with pytest.raises(ValueError):
        replay(START_FEN, ["a3b3"])


def test_history_plane_orientation_and_repetition():
    initial = encode_history([START_FEN])
    assert initial.shape == (124, 10, 9)
    assert initial[4, 0, 1] == 1  # red horse b0
    assert initial[11, 9, 1] == 1  # black horse b9
    assert initial[15:120].sum() == 0
    black_fen = play(START_FEN, "b0c2")
    black = encode_history([START_FEN, black_fen])
    assert black[4, 0, 1] == 1  # ours is now black, with ranks mirrored
    assert black[11, 7, 2] == 1  # red horse c2 -> rank7
    assert black[120].sum() == 90
    assert black[123].sum() == 90
    history = replay(START_FEN, ["b0c2", "b9c7", "c2b0", "c7b9"])
    repeated = encode_history(history)
    assert repeated[14].sum() == 90


def test_split_before_qa_and_future_answers():
    rows = make_records(games=10, plies=20)
    assert {r["stage"] for r in rows} == {"static_current", "dynamic_current", "static_future", "dynamic_future"}
    games = {s: {r["game_id"] for r in rows if r["split"] == s} for s in ["train", "validation"]}
    roots = {s: {" ".join(r["fen"].split()[:2]) for r in rows if r["split"] == s} for s in ["train", "validation"]}
    assert not games["train"] & games["validation"]
    assert not roots["train"] & roots["validation"]
    for row in rows[::9]:
        assert replay(row["initial_fen"], row["moves"])[-1] == row["fen"]
        replay(row["fen"], row["future_moves"])
