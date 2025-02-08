import random
import chess
import chess.svg
import numpy as np
from scipy.linalg import eig as dense_eig


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
###############################################################################
# 1) Piece weights and adjacency
###############################################################################
PIECE_WEIGHTS = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0  # set a value if you want to emphasize the king
}

def piece_weight(piece: chess.Piece) -> float:
    """
    Returns a signed weight for the given piece (+ for White, - for Black).
    """
    base_val = PIECE_WEIGHTS[piece.piece_type]
    return base_val if piece.color == chess.WHITE else -base_val

def build_adjacency_matrix(board: chess.Board) -> np.ndarray:
    """
    Builds a 64x64 adjacency matrix A for legal moves:
      A[i, j] = piece_weight if a piece on square i can legally move to j.
    """
    A = np.zeros((64, 64), dtype=np.float64)
    all_legal = list(board.legal_moves)
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        
        w = piece_weight(piece)
        # record for moves starting at 'sq'
        for mv in all_legal:
            if mv.from_square == sq:
                A[sq, mv.to_square] = w
    return A

###############################################################################
# 2) Spectral Graph Helpers
###############################################################################
def principal_eigenvector(A: np.ndarray):
    """
    Returns (lambda_max, v_max) for the largest eigenvalue (by real part).
    """
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx_max = np.argmax(vals)
    return vals[idx_max], vecs[:, idx_max]

def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    """
    L = D - A, where D is diag of row-sums in A.
    """
    degrees = np.sum(A, axis=1)
    return np.diag(degrees) - A

def fiedler_vector(L: np.ndarray):
    """
    Returns (lambda_2, vec_2), the second-smallest eigenvalue & vector of L.
    """
    vals, vecs = dense_eig(L)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx_sort = np.argsort(vals)
    f_idx = idx_sort[1]  # second smallest
    return vals[f_idx], vecs[:, f_idx]

def top_two_eigs(A: np.ndarray):
    """
    Returns ((lambda1, lambda2), (v1, v2)) for the top two eigenvalues (largest real parts).
    """
    vals, vecs = dense_eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    
    idx_sorted = np.argsort(vals)[::-1]  # descending order
    lam1, lam2 = vals[idx_sorted[0]], vals[idx_sorted[1]]
    v1, v2 = vecs[:, idx_sorted[0]], vecs[:, idx_sorted[1]]
    return (lam1, lam2), (v1, v2)

def sign_vector(vec: np.ndarray) -> np.ndarray:
    """
    Returns +1 / -1 / 0 for each component in vec.
    """
    out = np.zeros_like(vec)
    out[vec > 0] = 1
    out[vec < 0] = -1
    return out

###############################################################################
# 3) The 5 Spectral Metrics
###############################################################################
def directional_influence_metric(A: np.ndarray) -> float:
    """
    DIM: Cosine similarity between principal eigenvector and a ones-vector.
    """
    _, v1 = principal_eigenvector(A)
    v1_norm = np.linalg.norm(v1)
    if v1_norm == 0:
        return 0.0
    
    ref = np.ones_like(v1)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return 0.0
    
    return float(np.dot(v1, ref) / (v1_norm * ref_norm))

def strategic_partitioning_index(A: np.ndarray) -> float:
    """
    SPI: We'll measure how balanced the partition from the Fiedler vector is.
         ratio = min(pos_count, neg_count) / total
         => near 0.5 if balanced.
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
    DPA: correlation between the top two eigenvectors.
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    
    c = np.corrcoef(v1, v2)[0, 1]
    return float(c)

def positional_stability_contrast(A: np.ndarray) -> float:
    """
    PSC: Hamming distance of sign(v1) vs. sign(v2), 
         where (v1, v2) are the top two eigenvectors.
    """
    (_, _), (v1, v2) = top_two_eigs(A)
    s1 = sign_vector(v1)
    s2 = sign_vector(v2)
    return float(np.sum(s1 != s2))

def cross_connectivity_balance(A1: np.ndarray, A2: np.ndarray) -> float:
    """
    CCB: sqrt of sum of squared differences between sorted eigenvalues.
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
# 4) Random Board Generator
###############################################################################
# def random_board(num_pieces_white=8, num_pieces_black=8) -> chess.Board:
#     """
#     Creates a random board by placing up to 'num_pieces_white' White pieces 
#     and 'num_pieces_black' Black pieces on random squares. 
#     For simplicity, no special rules (like castling, en-passant).
#     """
#     board = chess.Board()
#     board.clear()
    
#     piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
#     squares = list(range(64))
#     random.shuffle(squares)
    
#     # Place white pieces
#     for _ in range(num_pieces_white):
#         if not squares:
#             break
#         sq = squares.pop()
#         p_type = random.choice(piece_types)
#         board.set_piece_at(sq, chess.Piece(p_type, chess.WHITE))
        
#     # Place black pieces
#     for _ in range(num_pieces_black):
#         if not squares:
#             break
#         sq = squares.pop()
#         p_type = random.choice(piece_types)
#         board.set_piece_at(sq, chess.Piece(p_type, chess.BLACK))
    
#     # Random side to move
#     board.turn = random.choice([chess.WHITE, chess.BLACK])
    
#     return board

###############################################################################
# 5) Distance Function
###############################################################################
def spectral_distance(board_a: chess.Board, board_b: chess.Board) -> float:
    """
    Example "distance" that compares the 5 metrics between board_a and board_b.
    
    1) We build adjacency for each.
    2) For the individual metrics (DIM, SPI, DPA, PSC), we compute them on each board 
       and sum up the absolute differences.
    3) For CCB, we compute cross_connectivity_balance(Aa, Ab) and add that in.
    
    You can adjust how you combine or weight these.
    """
    A_a = build_adjacency_matrix(board_a)
    A_b = build_adjacency_matrix(board_b)
    
    # Metrics for board A
    dim_a = directional_influence_metric(A_a)
    spi_a = strategic_partitioning_index(A_a)
    dpa_a = dynamic_potential_asymmetry(A_a)
    psc_a = positional_stability_contrast(A_a)
    
    # Metrics for board B
    dim_b = directional_influence_metric(A_b)
    spi_b = strategic_partitioning_index(A_b)
    dpa_b = dynamic_potential_asymmetry(A_b)
    psc_b = positional_stability_contrast(A_b)
    
    # Cross metric
    ccb_val = cross_connectivity_balance(A_a, A_b)
    
    # Summation of absolute differences for the 4 "individual" metrics
    diff_sum = abs(dim_a - dim_b) + abs(spi_a - spi_b) + abs(dpa_a - dpa_b) + abs(psc_a - psc_b)
    print("directional_influence_metric": dim_a, dim_b)
    print("strategic_partitioning_index": spi_a, spi_b)
    print("dynamic_potential_asymmetry": dpa_a, dpa_b)
    print("positional_stability_contrast": psc_a , psc_b)
    print("cross_connectivity_balance": ccb_val)
    
    # Combine them. 
    # This is arbitrary, so feel free to tweak weighting. 
    total_dist = diff_sum + ccb_val
    return total_dist

###############################################################################
# 6) Main Loop, Searching for Min Distance
###############################################################################
def main():
    min_dist = float("inf")
    
    # 1) Generate one static board (board_a)
    board_a = random_board(num_moves=20)
    
    print("Starting reference board (board_a):")
    print(board_a)
    
    # 2) Loop over multiple random boards (board_b)
    for i in range(100):
        board_b = random_board(num_moves=20)
        
        dist_val = spectral_distance(board_a, board_b)
        
        if dist_val < min_dist:
            min_dist = dist_val
            print(f"\n[Iteration {i+1}] New min dist found: {min_dist:.4f}")
            
            # Optionally save both boards as SVG
            svg_code_a = chess.svg.board(board=board_a, orientation=chess.WHITE)
            svg_code_b = chess.svg.board(board=board_b, orientation=chess.WHITE)
            with open("board_a.svg", "w", encoding="utf-8") as fa:
                fa.write(svg_code_a)
            with open("board_b.svg", "w", encoding="utf-8") as fb:
                fb.write(svg_code_b)
            
            print("Saved board_a.svg and board_b.svg")

if __name__ == "__main__":
    main()
