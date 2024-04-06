#pragma once

#include <vector>
#include "namespaces.hpp"

BEGIN_GAME_NAMESPACE

#ifndef BEGIN_TIC_TAC_TOE_NAMESPACE
#   define BEGIN_TIC_TAC_TOE_NAMESPACE namespace tic_tac_toe {
#endif

#ifndef END_TIC_TAC_TOE_NAMESPACE
#   define END_TIC_TAC_TOE_NAMESPACE }
#endif // END_TIC_TAC_TOE_NAMESPACE

BEGIN_TIC_TAC_TOE_NAMESPACE

enum class Player {
    NONE,
    X,
    O,
};

class engine {
public:
    engine() : x_bitboard(0), o_bitboard(0), board(9, 0),
     game_over(false), current_player(Player::X), winner(Player::NONE) {};
    ~engine() = default;

    float evaluate() const;
    void make_move(int move);
    void undo_move(int move);
    void validate();
    bool is_game_over() const;
    bool is_draw() const;
    bool is_winner() const;
    std::vector<int> get_moves() const;
    void set_current_player(Player player);

private:
    Player current_player;
    int x_bitboard;
    int o_bitboard;
    int *current_bitboard;
    std::vector<int> board;
    bool game_over;
    Player winner;
};


class TicTacToe {
public:
    TicTacToe() = default;
    ~TicTacToe() = default;
};

END_TIC_TAC_TOE_NAMESPACE



END_GAME_NAMESPACE