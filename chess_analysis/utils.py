# chess_analysis/utils.py
import os
import random
import chess
from chess_position import ChessPosition
from graph_construction import GraphConstructionFunctor

def is_good_position(position, min_pieces=5):
    board = position.board
    if board.is_game_over():
        return False
    piece_count = len(board.piece_map())
    if piece_count < min_pieces:
        return False
    cg = GraphConstructionFunctor("control")(position)
    if cg.number_of_nodes() == 0:
        return False
    return True

def generate_random_position(depth=10):
    board = chess.Board()
    for _ in range(depth):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return ChessPosition(board.fen())

def generate_random_positions(num_positions=50, depth=10, max_attempts=200):
    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < max_attempts:
        pos = generate_random_position(depth)
        if is_good_position(pos):
            positions.append(pos)
        attempts += 1
    if len(positions) < num_positions:
        print(f"Warning: Only generated {len(positions)} good positions out of requested {num_positions}.")
    return positions

def visualize_positions(position1, position2):
    import chess.svg
    svg1 = chess.svg.board(position1.board, size=300)
    svg2 = chess.svg.board(position2.board, size=300)
    return svg1, svg2

def normalize_eval(eval_result):
    try:
        if eval_result['type'] == 'cp':
            return eval_result['value'] / 100.0
        else:
            return 1.0 if eval_result['value'] > 0 else -1.0
    except Exception as e:
        print("Error normalizing evaluation:", e)
        return 0.0
