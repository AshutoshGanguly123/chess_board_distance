from feature_distance import demo_feature_distance
from jensen_shanon_distance import main_demo_info_distance
from physical_board_distance import demo_piecewise_distance
import chess
import random
import math
import chess.svg
import chess.engine

def random_board(num_moves=5):
    """
    Returns a board after making 'num_moves' random (legal) moves from the start,
    and then forcing it to be White's turn.
    """
    board = chess.Board()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    # Force White to move next
    board.turn = chess.WHITE
    return board


def check_pawn_structure(boardA: chess.Board, boardB: chess.Board):
    """
    Returns True if Board B's pawn structure does NOT violate the
    'no backward movement' rule for each color's pawns.
    
    - White pawns in B must be on same or higher rank than in A.
    - Black pawns in B must be on same or lower rank than in A.
    
    Otherwise returns False.
    """
    
    # 1. Gather White pawns
    white_pawns_A = sorted(boardA.pieces(chess.PAWN, chess.WHITE))
    white_pawns_B = sorted(boardB.pieces(chess.PAWN, chess.WHITE))
    
    # 2. Gather Black pawns
    black_pawns_A = sorted(boardA.pieces(chess.PAWN, chess.BLACK))
    black_pawns_B = sorted(boardB.pieces(chess.PAWN, chess.BLACK))
    
    # If the counts differ, decide how you want to handle it.
    # For now, let's only compare up to the min() of both sets.
    # The rest are effectively "unmatched" pawns.
    
    # White check
    n_white = min(len(white_pawns_A), len(white_pawns_B))
    for i in range(n_white):
        sqA = white_pawns_A[i]  # 0..63
        sqB = white_pawns_B[i]
        rankA, fileA = divmod(sqA, 8)  # rank = 0..7
        rankB, fileB = divmod(sqB, 8)
        
        # For White, we want rankB >= rankA
        if rankB < rankA:
            return False  # a white pawn has "gone backward"
    
    # Black check
    n_black = min(len(black_pawns_A), len(black_pawns_B))
    for i in range(n_black):
        sqA = black_pawns_A[i]
        sqB = black_pawns_B[i]
        rankA, fileA = divmod(sqA, 8)
        rankB, fileB = divmod(sqB, 8)
        
        # For Black, we want rankB <= rankA
        if rankB > rankA:
            return False  # a black pawn has "gone backward"
    
    # If we get here, there's no direct "backward movement" violation
    return True

min_dist = 10000
board_a = random_board(num_moves=18)
for _ in range(1000):
    board_b = random_board(num_moves=18)
    #if check_pawn_structure(board_a, board_b) == True:
    dist_feature = demo_feature_distance(board_a, board_b)/40
    dist_physical_board = demo_piecewise_distance(board_a, board_b)/10
    dist_jensen_shanon = main_demo_info_distance(board_a, board_b)
    total_dist = dist_feature + dist_physical_board + dist_jensen_shanon
    if min_dist > total_dist:
        min_dist = total_dist
        print("new min dist found", total_dist)
        svg_code_a = chess.svg.board(board=board_a, orientation=chess.WHITE)
        svg_code_b = chess.svg.board(board=board_b, orientation=chess.WHITE)
        with open("board_a.svg", "w") as f:
            f.write(svg_code_a)
        with open("board_b.svg", "w") as f:
            f.write(svg_code_b)