import random
import chess
import chess.svg
import numpy as np
from scipy.linalg import eig as dense_eig

###############################################################################
# A) Random Board Generator
###############################################################################
def random_board(num_moves=5):
    """
    Returns a board after making 'num_moves' random (legal) moves from the standard start.
    Then forces it to be White's turn for consistency.
    """
    board = chess.Board()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    board.turn = chess.WHITE
    return board

###############################################################################
# B) Piece Weights & Adjacency
###############################################################################
PIECE_WEIGHTS = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0  # or some larger weight if you want to emphasize the king
}

def piece_weight(piece: chess.Piece) -> float:
    """
    Returns signed weight for the piece: positive for White, negative for Black.
    """
    base_val = PIECE_WEIGHTS[piece.piece_type]
    return base_val if piece.color == chess.WHITE else -base_val

def build_adjacency_matrix_for_color(board: chess.Board, color: bool) -> np.ndarray:
    """
    Builds a 64x64 adjacency matrix for ONLY the given color's pieces.
    - color=True => White
    - color=False => Black
    We only consider moves by that color's pieces.
    
    A[i,j] = piece_weight if a piece (of the specified color) on square i can move to j.
    """
    A = np.zeros((64, 64), dtype=np.float64)
    
    # We only want to look at legal moves for that color.
    # Let's force 'board.turn = color' temporarily (in a copy).
    board_copy = board.copy()
    board_copy.turn = color
    
    all_legal = list(board_copy.legal_moves)
    
    for sq in range(64):
        piece = board_copy.piece_at(sq)
        if piece is None:
            continue
        if piece.color != color:
            continue  # skip pieces of the other color
        
        w = piece_weight(piece)
        for mv in all_legal:
            if mv.from_square == sq:
                A[sq, mv.to_square] = w
    
    return A

def build_adjacency_matrix_entire(board: chess.Board) -> np.ndarray:
    """
    Builds a 64x64 adjacency matrix for the entire board (both colors).
    This is used for the Cross-Connectivity Balance (CCB) metric.
    """
    A = np.zeros((64, 64), dtype=np.float64)
    # Use the board's actual side to move for generating legal moves
    all_legal = list(board.legal_moves)
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        w = piece_weight(piece)
        for mv in all_legal:
            if mv.from_square == sq:
                A[sq, mv.to_square] = w
    return A

###############################################################################
# C) Spectral Graph Helpers
###############################################################################
def principal_eigenvector(A: np.ndarray):
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx_max = np.argmax(vals)
    return vals[idx_max], vecs[:, idx_max]

def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    degrees = np.sum(A, axis=1)
    return np.diag(degrees) - A

def fiedler_vector(L: np.ndarray):
    vals, vecs = dense_eig(L)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx_sorted = np.argsort(vals)
    # second smallest
    if len(idx_sorted) > 1:
        f_idx = idx_sorted[1]
    else:
        # degenerate case => return 0's
        return 0.0, np.zeros_like(vals)
    return vals[f_idx], vecs[:, f_idx]

def top_two_eigs(A: np.ndarray):
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    
    idx_sorted = np.argsort(vals)[::-1]
    if len(idx_sorted) < 2:
        # If there's only 0 or 1 eigenvalue effectively, handle degenerate
        lam1 = vals[idx_sorted[0]] if len(idx_sorted) > 0 else 0.0
        lam2 = 0.0
        v1 = vecs[:, idx_sorted[0]] if len(idx_sorted) > 0 else np.zeros(len(vals))
        v2 = np.zeros(len(vals))
        return (lam1, lam2), (v1, v2)
    
    lam1 = vals[idx_sorted[0]]
    lam2 = vals[idx_sorted[1]]
    v1 = vecs[:, idx_sorted[0]]
    v2 = vecs[:, idx_sorted[1]]
    return (lam1, lam2), (v1, v2)

def sign_vector(vec: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vec)
    out[vec > 0] = 1
    out[vec < 0] = -1
    return out

###############################################################################
# D) Four Metrics (per color)
###############################################################################
def directional_influence_metric(A: np.ndarray) -> float:
    """
    DIM = cos_sim(principal_eigenvector, ones-vector).
    """
    _, v1 = principal_eigenvector(A)
    v1_norm = np.linalg.norm(v1)
    if v1_norm == 0:
        return 0.0
    ref = np.ones_like(v1)
    r_norm = np.linalg.norm(ref)
    if r_norm == 0:
        return 0.0
    return float(np.dot(v1, ref) / (v1_norm * r_norm))

def strategic_partitioning_index(A: np.ndarray) -> float:
    """
    SPI: measures how 'balanced' the sign-based Fiedler partition is.
    ratio = min(#pos, #neg)/ (total)
    """
    L = laplacian_matrix(A)
    _, f = fiedler_vector(L)
    pos_count = np.sum(f >= 0)
    neg_count = np.sum(f < 0)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return float(min(pos_count, neg_count) / total)

def dynamic_potential_asymmetry(A: np.ndarray) -> float:
    """
    DPA: correlation between top two eigenvectors (v1, v2).
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    c = np.corrcoef(v1, v2)[0, 1]
    return float(c)

def positional_stability_contrast(A: np.ndarray) -> float:
    """
    PSC: Hamming distance of sign(v1) vs. sign(v2).
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    s1 = sign_vector(v1)
    s2 = sign_vector(v2)
    return float(np.sum(s1 != s2))

###############################################################################
# E) Fifth Metric: Cross Connectivity (whole board)
###############################################################################
def cross_connectivity_balance(A1: np.ndarray, A2: np.ndarray) -> float:
    """
    CCB: sqrt(sum((sorted_eigs(A1) - sorted_eigs(A2))^2)).
    """
    vals1, _ = dense_eig(A1)
    vals2, _ = dense_eig(A2)
    vals1 = np.real(vals1)
    vals2 = np.real(vals2)
    s1 = np.sort(vals1)
    s2 = np.sort(vals2)
    min_len = min(len(s1), len(s2))
    dist_sq = np.sum((s1[:min_len] - s2[:min_len])**2)
    return float(np.sqrt(dist_sq))

###############################################################################
# F) Main Distance Computation
###############################################################################
def spectral_distance(board_a: chess.Board, board_b: chess.Board) -> float:
    """
    1) Build adjacency for White of board_a, White of board_b => compute 4 metrics
       (DIM, SPI, DPA, PSC), take sum of absolute differences.
    2) Build adjacency for Black of board_a, Black of board_b => compute 4 metrics,
       sum of absolute differences.
    3) Build adjacency for entire board_a, entire board_b => CCB => add to the total.
    """
    # --- White adjacency
    A_a_white = build_adjacency_matrix_for_color(board_a, chess.WHITE)
    A_b_white = build_adjacency_matrix_for_color(board_b, chess.WHITE)
    
    # compute the 4 metrics for White on board_a
    dim_a_w = directional_influence_metric(A_a_white)
    spi_a_w = strategic_partitioning_index(A_a_white)
    dpa_a_w = dynamic_potential_asymmetry(A_a_white)
    psc_a_w = positional_stability_contrast(A_a_white)
    
    # compute the 4 metrics for White on board_b
    dim_b_w = directional_influence_metric(A_b_white)
    spi_b_w = strategic_partitioning_index(A_b_white)
    dpa_b_w = dynamic_potential_asymmetry(A_b_white)
    psc_b_w = positional_stability_contrast(A_b_white)
    
    # Summation of absolute diffs for White
    white_diff = (abs(dim_a_w - dim_b_w) + 
                  abs(spi_a_w - spi_b_w) +
                  abs(dpa_a_w - dpa_b_w) +
                  abs(psc_a_w - psc_b_w))
    
    # --- Black adjacency
    A_a_black = build_adjacency_matrix_for_color(board_a, chess.BLACK)
    A_b_black = build_adjacency_matrix_for_color(board_b, chess.BLACK)
    
    # compute the 4 metrics for Black on board_a
    dim_a_b = directional_influence_metric(A_a_black)
    spi_a_b = strategic_partitioning_index(A_a_black)
    dpa_a_b = dynamic_potential_asymmetry(A_a_black)
    psc_a_b = positional_stability_contrast(A_a_black)
    
    # compute the 4 metrics for Black on board_b
    dim_b_b = directional_influence_metric(A_b_black)
    spi_b_b = strategic_partitioning_index(A_b_black)
    dpa_b_b = dynamic_potential_asymmetry(A_b_black)
    psc_b_b = positional_stability_contrast(A_b_black)
    
    # Summation of absolute diffs for Black
    black_diff = (abs(dim_a_b - dim_b_b) +
                  abs(spi_a_b - spi_b_b) +
                  abs(dpa_a_b - dpa_b_b) +
                  abs(psc_a_b - psc_b_b))
    
    # --- Whole-board adjacency for CCB
    A_a_entire = build_adjacency_matrix_entire(board_a)
    A_b_entire = build_adjacency_matrix_entire(board_b)
    ccb_val = cross_connectivity_balance(A_a_entire, A_b_entire)
    
    # Combined distance
    total_dist = white_diff + black_diff + ccb_val
    
    print("\n--- Spectral Distances Debug ---")
    print(f"White Metrics (Board A vs Board B):")
    print(f"  DIM A={dim_a_w:.4f}, DIM B={dim_b_w:.4f}")
    print(f"  SPI A={spi_a_w:.4f}, SPI B={spi_b_w:.4f}")
    print(f"  DPA A={dpa_a_w:.4f}, DPA B={dpa_b_w:.4f}")
    print(f"  PSC A={psc_a_w:.4f}, PSC B={psc_b_w:.4f}")
    print(f"  => White diff: {white_diff:.4f}")
    
    print(f"Black Metrics (Board A vs Board B):")
    print(f"  DIM A={dim_a_b:.4f}, DIM B={dim_b_b:.4f}")
    print(f"  SPI A={spi_a_b:.4f}, SPI B={spi_b_b:.4f}")
    print(f"  DPA A={dpa_a_b:.4f}, DPA B={dpa_b_b:.4f}")
    print(f"  PSC A={psc_a_b:.4f}, PSC B={psc_b_b:.4f}")
    print(f"  => Black diff: {black_diff:.4f}")
    
    print(f"CCB (entire board): {ccb_val:.4f}")
    print(f"TOTAL DIST = {total_dist:.4f}")
    
    return total_dist

###############################################################################
# G) Main
###############################################################################
def main():
    min_dist = float("inf")
    
    # Generate one static board (board_a)
    board_a = random_board(num_moves=20)
    print("Reference Board A:\n", board_a)
    
    # Generate many candidates
    for i in range(100):
        board_b = random_board(num_moves=20)
        
        dist_val = spectral_distance(board_a, board_b)
        
        if dist_val < min_dist:
            min_dist = dist_val
            print(f"\n[Iteration {i+1}] *** New min distance found: {min_dist:.4f} ***")
            
            # Save boards as SVG
            svg_a = chess.svg.board(board=board_a, orientation=chess.WHITE)
            svg_b = chess.svg.board(board=board_b, orientation=chess.WHITE)
            with open("board_a.svg", "w", encoding="utf-8") as fa:
                fa.write(svg_a)
            with open("board_b.svg", "w", encoding="utf-8") as fb:
                fb.write(svg_b)
            print("Saved board_a.svg and board_b.svg")

if __name__ == "__main__":
    main()
