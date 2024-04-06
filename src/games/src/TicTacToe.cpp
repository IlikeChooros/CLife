#include <games/games.hpp>

#include <algorithm>

BEGIN_GAME_NAMESPACE

BEGIN_TIC_TAC_TOE_NAMESPACE

float engine::evaluate() const {
    int bitboards[] = {
        0b111000000, 0b000111000, 0b000000111, // rows
        0b100100100, 0b010010010, 0b001001001, // cols
        0b100010001, 0b001010100 // diagonals
    };
    
    return is_game_over() ? (is_winner() ? 1.0f : -1.0f) : 0.f;
}

void engine::set_current_player(Player player) {
    current_player = player;
    current_bitboard = current_player == Player::X ? &x_bitboard : &o_bitboard;
}

void engine::make_move(int move) {
    board[move] = static_cast<int>(current_player);
    *current_bitboard |= 1 << move;
}

void engine::undo_move(int move) {
    board[move] = 0;
    *current_bitboard ^= 1 << move;
}

void engine::validate() {
    int bitboards[] = {
        0b111000000, 0b000111000, 0b000000111, // rows
        0b100100100, 0b010010010, 0b001001001, // cols
        0b100010001, 0b001010100 // diagonals
    };
    int player_bitboards[] = {x_bitboard, o_bitboard};
    Player players[] = {Player::X, Player::O};

    for (int j = 0; j < 2; j++){
        for(int i = 0; i < 8; i++){
            if((player_bitboards[j] & bitboards[i]) == bitboards[i]){
                game_over = true;
                winner = players[j];
                return;
            }
        }
    }
}

bool engine::is_draw() const {
    return (o_bitboard | x_bitboard == 0b111111111) && !game_over;
}

bool engine::is_winner() const {
    return game_over && winner == current_player;
}

bool engine::is_game_over() const {
    return game_over;
}

std::vector<int> engine::get_moves() const{
    std::vector<int> moves;
    for(int i = 0; i < 9; i++){
        if(board[i] == 0){
            moves.push_back(i);
        }
    }
    return moves;
}

END_TIC_TAC_TOE_NAMESPACE

END_GAME_NAMESPACE