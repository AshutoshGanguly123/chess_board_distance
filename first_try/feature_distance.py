import chess
import random
import math
import chess.svg
import chess.engine

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # We'll ignore the king's 'value' in terms of material
}

def squares_attacked(board: chess.Board, color: bool) -> int:
    """
    Counts how many squares on the board are attacked by the given color.
    """
    count = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            count += 1
    return count

def pieces_attacked(board: chess.Board, color: bool) -> int:
    """
    Counts how many pieces of 'color' are attacked by the *opposite* color.
    """
    opp_color = not color
    count = 0
    for square, piece in board.piece_map().items():
        if piece.color == color:
            # If this piece is attacked by the opponent
            if board.is_attacked_by(opp_color, square):
                count += 1
    return count

def squares_around_king(board: chess.Board, color: bool):
    """
    Returns a list of squares immediately surrounding the king of the given color.
    If the king is not on the board (extremely rare), returns an empty list.
    """
    king_square = board.king(color)
    if king_square is None:
        return []
    # Use precomputed bitboard for king moves.
    return list(chess.SquareSet(chess.BB_KING_ATTACKS[king_square]))

def king_unsafe(board: chess.Board, color: bool) -> int:
    """
    Measures how "unsafe" the king is by counting how many squares around
    the king are attacked by the opponent.
    """
    opp_color = not color
    king_zone = squares_around_king(board, color)
    return sum(1 for sq in king_zone if board.is_attacked_by(opp_color, sq))

def degree_of_freedom(board: chess.Board, color: bool) -> int:
    """
    Counts how many moves 'color' could make if it were that color's turn,
    regardless of whose turn it actually is.
    """
    board_copy = board.copy()
    board_copy.turn = color
    return board_copy.legal_moves.count()

def extract_features(board: chess.Board):
    """
    Returns a more 'dense' list of numeric features representing the board.
    """

    # 1. Material Balance
    white_material = 0
    black_material = 0
    for piece_type, value in PIECE_VALUES.items():
        white_material += len(board.pieces(piece_type, chess.WHITE)) * value
        black_material += len(board.pieces(piece_type, chess.BLACK)) * value
    material_balance = white_material - black_material

    # 2. Total Pieces
    total_pieces = len(board.piece_map())

    # 3. Number of Moves Available (for whoever is to move right now)
    moves_available = board.legal_moves.count()

    # 4. Squares Attacked by White vs. Black
    white_attacked_squares = squares_attacked(board, chess.WHITE)
    black_attacked_squares = squares_attacked(board, chess.BLACK)
    attacked_squares_diff = white_attacked_squares - black_attacked_squares

    # 5. Pieces Attacked (White vs. Black)
    white_pieces_under_attack = pieces_attacked(board, chess.WHITE)
    black_pieces_under_attack = pieces_attacked(board, chess.BLACK)
    pieces_attacked_diff = white_pieces_under_attack - black_pieces_under_attack

    # 6. King Safety (count how many squares around each king are attacked)
    white_king_unsafe = king_unsafe(board, chess.WHITE)
    black_king_unsafe = king_unsafe(board, chess.BLACK)
    # Define a difference (positive means black king is more in danger):
    king_safety_diff = black_king_unsafe - white_king_unsafe

    # 7. Degree of Freedom (sum of possible moves for White vs. Black)
    white_dof = degree_of_freedom(board, chess.WHITE)
    black_dof = degree_of_freedom(board, chess.BLACK)
    dof_diff = white_dof - black_dof

    # Return them all as a feature vector
    return [
        material_balance,       # 0
        total_pieces,           # 1
        moves_available,        # 2
        attacked_squares_diff,  # 3
        pieces_attacked_diff,   # 4
        king_safety_diff,       # 5
        dof_diff                # 6
    ]

def feature_distance(board1: chess.Board, board2: chess.Board):
    """
    Computes the Euclidean distance between two boards in the new feature space.
    """
    f1 = extract_features(board1)
    f2 = extract_features(board2)
    p = 2
    return sum(abs(x - y)**p for x, y in zip(f1, f2))**(1.0 / p)

# def random_board(num_moves=5):
#     """
#     Returns a board after making 'num_moves' random (legal) moves from the start,
#     and then forcing it to be White's turn.
#     """
#     board = chess.Board()
#     for _ in range(num_moves):
#         if board.is_game_over():
#             break
#         move = random.choice(list(board.legal_moves))
#         board.push(move)
#     # Force White to move next
#     board.turn = chess.WHITE
#     return board

#engine eval
def get_engine_eval(board: chess.Board, depth=15) -> float:
    
    engine_path = "/opt/homebrew/bin/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))    
    # Convert the engine's evaluation to a centipawn score from White's POV
    score = info["score"].white().score(mate_score=100000)
    return score


def demo_feature_distance(board_a, board_b):    
    # Compute distance
    dist = feature_distance(board_a, board_b)
    #print("Distance (feature-based):", dist)

    #eval_a = get_engine_eval(board_a, depth=15)
    #eval_b = get_engine_eval(board_b, depth=15)

    #eval_diff = abs(eval_a - eval_b)
    #print(f"Difference in engine eval = {eval_diff:.1f}")
    #print(eval_a)
    #print(eval_b)

    # Generate SVG code for final boards, ensuring White is at bottom
    # svg_code_a = chess.svg.board(board=board_a, orientation=chess.WHITE)
    # svg_code_b = chess.svg.board(board=board_b, orientation=chess.WHITE)
    # with open("board_a.svg", "w") as f:
    #     f.write(svg_code_a)
    # with open("board_b.svg", "w") as f:
    #     f.write(svg_code_b)
    return dist

# if __name__ == "__main__":
#     demo_feature_distance()
