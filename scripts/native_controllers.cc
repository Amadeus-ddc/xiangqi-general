// Read geometric controllers from unmodified, pinned Pikafish rule code.
#include <iostream>
#include <string>

#include "attacks.h"
#include "position.h"

int main() {
    using namespace Stockfish;
    Attacks::init();
    Position::init();
    std::string fen;
    while (std::getline(std::cin, fen)) {
        if (fen.empty())
            continue;
        // Translate only placement aliases, leaving the black-to-move field intact.
        const auto placement_end = fen.find(' ');
        if (placement_end == std::string::npos) {
            std::cerr << "Expected a complete FEN" << '\n';
            return 2;
        }
        for (std::size_t i = 0; i < placement_end; ++i) {
            if (fen[i] == 'h') fen[i] = 'n';
            if (fen[i] == 'H') fen[i] = 'N';
            if (fen[i] == 'e') fen[i] = 'b';
            if (fen[i] == 'E') fen[i] = 'B';
        }
        Position pos;
        StateInfo state;
        if (auto error = pos.set(fen, &state)) {
            std::cerr << error->what() << '\n';
            return 2;
        }
        for (Square square = SQ_A0; square <= SQ_I9; ++square) {
            std::cout << char('a' + file_of(square)) << int(rank_of(square)) << ':';
            Bitboard controllers = pos.attackers_to(square);
            bool first = true;
            while (controllers) {
                const Square source = pop_lsb(controllers);
                if (!first) std::cout << ',';
                std::cout << char('a' + file_of(source)) << int(rank_of(source));
                first = false;
            }
            std::cout << '\n';
        }
    }
}
