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

for _ in range(1000):
    board_a = random_board(num_moves=30)
    board_b = random_board(num_moves=30)

    dist_jensen_shanon = demo_piecewise_distance(board_a, board_b)

    print(dist_jensen_shanon)


# dist_jensen_shanon = demo_piecewise_distance(board_a, board_b)


# svg_code_a = chess.svg.board(board=board_a, orientation=chess.WHITE)
# svg_code_b = chess.svg.board(board=board_b, orientation=chess.WHITE)
# with open("board_a.svg", "w") as f:
#     f.write(svg_code_a)
# with open("board_b.svg", "w") as f:
#     f.write(svg_code_b)