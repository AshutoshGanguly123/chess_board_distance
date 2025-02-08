import chess
import random
import math
import chess.svg
import heapq

# 1) Move cost function
def move_cost(board, move):
    temp_board = board.copy()
    is_capture = temp_board.is_capture(move)
    promo = (move.promotion is not None)
    is_castle = temp_board.is_castling(move)

    cost = 1.0  # base cost for quiet move
    if is_capture:
        cost += 1.0
    if promo:
        cost += 1.0
    if is_castle:
        cost += 0.5

    return cost

# 2) Position signature
def position_signature(board):
    return board.fen()

# 3) Weighted move distance (Dijkstra with depth limit)
def weighted_move_distance(board1, board2, max_depth=4):
    start_fen = position_signature(board1)
    goal_fen = position_signature(board2)
    if start_fen == goal_fen:
        return 0.0

    # Priority queue: (cost_so_far, depth, board)
    pq = [(0.0, 0, board1)]
    visited_cost = {start_fen: 0.0}

    while pq:
        current_cost, depth, current_board = heapq.heappop(pq)
        current_fen = position_signature(current_board)

        if current_fen == goal_fen:
            return current_cost

        if depth < max_depth:
            for move in current_board.legal_moves:
                move_c = move_cost(current_board, move)

                next_board = current_board.copy()
                next_board.push(move)
                next_fen = position_signature(next_board)

                new_cost = current_cost + move_c
                if next_fen not in visited_cost or new_cost < visited_cost[next_fen]:
                    visited_cost[next_fen] = new_cost
                    heapq.heappush(pq, (new_cost, depth + 1, next_board))

    return None  # Not found within max_depth

# 4) Helper function to create a random board
def random_board(num_moves=5):
    board = chess.Board()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    # Force White to move next (optional, if we like White’s perspective)
    board.turn = chess.WHITE
    return board

# 5) Test / Demo
def demo_weighted_move_distance():
    board_a = random_board(num_moves=3)
    board_b = random_board(num_moves=3)

    # Show the FENs and Weighted Move Distance
    print("Board A:", board_a.fen())
    print("Board B:", board_b.fen())

    dist = weighted_move_distance(board_a, board_b, max_depth=4)
    print("Weighted Move Distance:", dist)

    # Save SVG
    svg_a = chess.svg.board(board_a, orientation=chess.WHITE)
    svg_b = chess.svg.board(board_b, orientation=chess.WHITE)
    with open("wmd_board_a.svg", "w") as f:
        f.write(svg_a)
    with open("wmd_board_b.svg", "w") as f:
        f.write(svg_b)

if __name__ == "__main__":
    demo_weighted_move_distance()
