// Independent input reference: call the unmodified official Px0 board/encoder.
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  using namespace lczero;
  InitializeMagicBitboards();
  ChessBoard board;
  int rule50 = 0, move_number = 1;
  board.SetFromFen(argv[1], &rule50, &move_number);
  PositionHistory history;
  history.Reset(board, rule50, 2 * (move_number - 1) + int(board.flipped()));
  for (int i = 2; i < argc; ++i) {
    std::string text(argv[i]);
    if (text.size() != 4) return 3;
    Move move = Move::White(Square::Parse(text.substr(0, 2)), Square::Parse(text.substr(2, 2)));
    if (history.IsBlackToMove()) move.Flip();
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), move) == legal.end()) return 4;
    history.Append(move);
  }
  auto planes = EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                                    history, 8, FillEmptyHistory::FEN_ONLY, nullptr);
  for (const auto& plane : planes) {
    for (int square = 0; square < 90; ++square) {
      float value = ((plane.mask >> square) & 1) ? plane.value : 0.0f;
      std::fwrite(&value, sizeof(value), 1, stdout);
    }
  }
  return 0;
}
