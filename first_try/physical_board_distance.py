import chess
import random
import math
from collections import deque

# --- Offsets ---
KNIGHT_OFFSETS = [
    (-2, -1), (-2, +1), (+2, -1), (+2, +1),
    (-1, -2), (-1, +2), (+1, -2), (+1, +2)
]
KING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, +1),
    (0, -1),          (0, +1),
    (1, -1),  (1, 0),  (1, +1)
]

def generate_offsets_rook():
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]

def generate_offsets_bishop():
    return [(1, 1), (1, -1), (-1, 1), (-1, -1)]

def generate_offsets_queen():
    return generate_offsets_rook() + generate_offsets_bishop()

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def square_to_coords(sq: chess.Square):
    # (rank, file), each 0..7
    return divmod(sq, 8)

# --- BFS for distances on an empty board ---
def bfs_min_steps(start_sq: chess.Square, end_sq: chess.Square, offsets, can_repeat=True):
    if start_sq == end_sq:
        return 0

    start_r, start_c = square_to_coords(start_sq)
    end_r, end_c = square_to_coords(end_sq)

    visited = set([(start_r, start_c)])
    queue = deque([(start_r, start_c, 0)])

    while queue:
        r, c, dist = queue.popleft()
        for (dr, dc) in offsets:
            nr, nc = r + dr, c + dc
            while in_bounds(nr, nc):
                if (nr, nc) == (end_r, end_c):
                    return dist + 1
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
                if not can_repeat:
                    break
                nr += dr
                nc += dc
    return 64  # Some large fallback

def bfs_min_steps_knight(start_sq, end_sq):
    if start_sq == end_sq:
        return 0
    start_r, start_c = square_to_coords(start_sq)
    end_r, end_c = square_to_coords(end_sq)

    visited = set([(start_r, start_c)])
    queue = deque([(start_r, start_c, 0)])
    while queue:
        r, c, dist = queue.popleft()
        for (dr, dc) in KNIGHT_OFFSETS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                if (nr, nc) == (end_r, end_c):
                    return dist + 1
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
    return 64

# --- Piece Weights (for final weighted average) ---
# You can tune the King’s “weight” if you wish—traditionally King = infinite in real chess,
# but we’ll treat it as 10 for the sake of relative weighting.
PIECE_WEIGHTS = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:  10
}

def piece_movement_distance(piece_type: chess.PieceType, from_sq: chess.Square, to_sq: chess.Square) -> int:
    """
    Returns the number of single-square steps (or knight jumps) needed
    for this piece type to go from 'from_sq' to 'to_sq' on an empty board.
    Pawns: we do the simplified 'same file => rank distance' approach.
    """
    if from_sq == to_sq:
        return 0

    # Pawn special handling (requested simplification)
    if piece_type == chess.PAWN:
        from_rank, from_file = square_to_coords(from_sq)
        to_rank, to_file = square_to_coords(to_sq)
        if from_file != to_file:
            # If different file, big penalty or sum of file+rank diff
            return abs(from_file - to_file) + abs(from_rank - to_rank)
        else:
            return abs(to_rank - from_rank)

    # Knight
    if piece_type == chess.KNIGHT:
        return bfs_min_steps_knight(from_sq, to_sq)

    # King
    if piece_type == chess.KING:
        return bfs_min_steps(from_sq, to_sq, KING_OFFSETS, can_repeat=False)

    # Bishop
    if piece_type == chess.BISHOP:
        return bfs_min_steps(from_sq, to_sq, generate_offsets_bishop(), can_repeat=True)

    # Rook
    if piece_type == chess.ROOK:
        return bfs_min_steps(from_sq, to_sq, generate_offsets_rook(), can_repeat=True)

    # Queen
    if piece_type == chess.QUEEN:
        return bfs_min_steps(from_sq, to_sq, generate_offsets_queen(), can_repeat=True)

    # Default
    return 0

def piecewise_distance(board1: chess.Board, board2: chess.Board) -> float:
    """
    Computes a weighted average of piecewise "movement distance."
    For each (color, piece_type), we pair up pieces, compute BFS-based distance,
    multiply by piece weight, and sum. Then we divide by the sum of piece weights
    to get an average.
    """
    total_weighted_dist = 0.0
    total_weights = 0.0

    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(chess.PAWN, chess.KING + 1):
            squares1 = sorted(board1.pieces(piece_type, color))
            squares2 = sorted(board2.pieces(piece_type, color))

            w = PIECE_WEIGHTS[piece_type]  # weight for this piece type
            # Pair them
            n = min(len(squares1), len(squares2))
            for i in range(n):
                sq1 = squares1[i]
                sq2 = squares2[i]
                dist = piece_movement_distance(piece_type, sq1, sq2)
                # Weighted by piece value
                total_weighted_dist += dist * w
                total_weights += w

            # Unmatched pieces penalty
            diff = abs(len(squares1) - len(squares2))
            # Each unmatched piece adds a penalty (e.g., 3 steps) times its piece weight
            penalty_distance = 3.0
            total_weighted_dist += diff * penalty_distance * w
            total_weights += diff * w

    if total_weights == 0:
        return 0.0
    # Weighted average
    return total_weighted_dist / total_weights

# --- Demo code ---
def random_board(num_moves=5):
    board = chess.Board()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    board.turn = chess.WHITE
    return board

def demo_piecewise_distance(board_a, board_b):
    # board_a = random_board(num_moves=5)
    # board_b = random_board(num_moves=5)

    dist = piecewise_distance(board_a, board_b)
    # print("Board A:", board_a)
    # print("Board B:", board_b)
    # print("Weighted avg piecewise distance =", dist)
    return dist
# if __name__ == "__main__":
#     demo_piecewise_distance()
